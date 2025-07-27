#!/usr/bin/env python3
"""Claude Code Hook: Token Consumption Tracker.

===========================================

This hook analyzes conversation files and generates token consumption reports.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def load_env_file(env_path=".env"):
    """Simple .env file loader using only built-in Python modules."""
    env_vars = {}
    env_file = Path(env_path)

    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip("\"'")
                    env_vars[key.strip()] = value

    return env_vars


# Configuration - easily changeable report settings
REPORT_FILENAME = "token_report.txt"
REPORT_PATH = ".claude/"  # Current directory, can be changed to absolute path

# Report template
REPORT_TEMPLATE = """============================================================
CLAUDE CONVERSATION TOKEN REPORT
============================================================
Generated: {{ generation_time }}
Conversation File: {{ conversation_file }}
Generation Took: {{ processing_time }}

TOKENS VS. CONTEXT WINDOW
--------------------
Context-Aware Tokens ({{ io_token_count }}) - Segment {{ current_segment }}
Context-aware = Cumulative I/O + Cache Read within current compacting segment
{{ context_window_labels }}
{{ context_window_meter }}

SUMMARY
--------------------
Total Messages: {{ total_messages }}
  - User Messages: {{ user_messages }}
  - Assistant Messages: {{ assistant_messages }}
  - Other Messages: {{ other_messages }}

Note: Token tracking data comes from Assistant messages only.

TOKEN CONSUMPTION
--------------------
Cumulative Input Tokens:        {{ cumulative_input_tokens }}
Cumulative Output Tokens:       {{ cumulative_output_tokens }}
Total I/O Tokens:               {{ total_io_tokens }}
Latest Cache Creation Tokens:   {{ latest_cache_creation_tokens }}
Latest Cache Read Tokens:       {{ latest_cache_read_tokens }}

{{ cache_efficiency_section }}

{{ horizontal_chart }}

{{ vertical_chart }}

============================================================"""


def find_current_conversation_file(conversation_path):
    """Find the most recent conversation file for this project."""
    # Look in the project's conversation directory
    project_dir = Path(conversation_path)
    if not project_dir.exists():
        return None

    # Find all JSONL files and get the most recent one
    jsonl_files = list(project_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    # Return the most recently modified file
    return max(jsonl_files, key=lambda f: f.stat().st_mtime)


def parse_conversation_tokens(jsonl_file):
    """Parse token usage from conversation JSONL file with compacting detection."""
    if not jsonl_file or not jsonl_file.exists():
        return None

    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    latest_cache_creation_tokens = 0
    latest_cache_read_tokens = 0
    message_count = 0
    assistant_messages = 0
    user_messages = 0

    # Track token messages with assistant message numbering and compacting segments
    token_messages = []
    assistant_token_counter = 0  # Sequential counter for assistant messages with tokens
    compacting_segment = 0  # Track compacting segments

    try:
        with open(jsonl_file, encoding="utf-8") as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())

                # Skip summary lines early
                if data.get("type") == "summary":
                    continue

                # Check for compacting events (parentUuid is null)
                if data.get("parentUuid") is None and line_num > 1:  # Skip first line (conversation start)
                    compacting_segment += 1

                message_count += 1

                # Count message types
                msg_type = data.get("type", "unknown")
                if msg_type == "assistant":
                    assistant_messages += 1
                elif msg_type == "user":
                    user_messages += 1

                # Only process assistant messages for token tracking
                if msg_type != "assistant":
                    continue

                # Extract usage data - all assistant messages have usage
                message = data.get("message", {})
                usage = message.get("usage", {})

                if not usage:
                    continue

                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_creation = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)

                # Include all messages with usage data (including zero tokens)
                assistant_token_counter += 1  # Increment assistant message counter

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cache_creation_tokens += cache_creation
                total_cache_read_tokens += cache_read

                # Track latest cache values (from most recent message)
                latest_cache_creation_tokens = cache_creation
                latest_cache_read_tokens = cache_read

                # Store token message with assistant message numbering and compacting segment
                token_messages.append(
                    {
                        "line": line_num,
                        "message_index": assistant_token_counter,
                        "compacting_segment": compacting_segment,
                        "type": msg_type,
                        "timestamp": data.get("timestamp", ""),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation": cache_creation,
                        "cache_read": cache_read,
                        "model": message.get("model", "unknown"),
                        "service_tier": usage.get("service_tier", "unknown"),
                    }
                )

            except json.JSONDecodeError:
                continue

    except Exception:
        return None

    return {
        "file_path": str(jsonl_file),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_creation_tokens": total_cache_creation_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "latest_cache_creation_tokens": latest_cache_creation_tokens,
        "latest_cache_read_tokens": latest_cache_read_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "message_count": message_count,
        "assistant_messages": assistant_messages,
        "user_messages": user_messages,
        "conversation_data": token_messages,
        "compacting_segments": compacting_segment + 1,  # Total number of segments
    }


def calculate_cumulative_tokens(conversation_data):
    """Pre-calculate context-aware token data for reuse across all charts."""
    if not conversation_data or len(conversation_data) < 2:
        return None, None, 0

    # Use centralized context-aware calculation
    context_aware_data, latest_tokens, _ = calculate_context_aware_tokens(conversation_data)
    
    if not context_aware_data:
        return None, None, 0
    
    # Extract context-aware tokens in message order
    context_aware_tokens = [item["context_aware_tokens"] for item in context_aware_data]
    max_tokens = max(context_aware_tokens) if context_aware_tokens else 0
    
    return conversation_data, context_aware_tokens, max_tokens


def generate_ascii_chart(conversation_data, width=50):
    """Generate ASCII chart showing context-aware token progression over time."""
    token_messages, context_aware_tokens, max_tokens = calculate_cumulative_tokens(
        conversation_data
    )
    if not token_messages:
        return ""

    chart = []
    chart.append("TOKEN PROGRESSION CHART (Horizontal)")
    chart.append("-" * 40)
    chart.append(f"Context-Aware Growth: 0 → {max_tokens:,} tokens")
    chart.append("")

    # Create ASCII chart - use last 10 data points with actual message indices
    display_count = min(10, len(context_aware_tokens))
    display_data = context_aware_tokens[-display_count:]
    display_messages = token_messages[-display_count:]

    chart_lines = []
    for _, (total, msg_data) in enumerate(
        zip(display_data, display_messages, strict=False)
    ):
        bar_length = int((total / max_tokens) * width)
        bar = "█" * bar_length + "░" * (width - bar_length)
        actual_msg_num = msg_data["message_index"]  # Use actual conversation position
        chart_lines.append(f"Msg {actual_msg_num:3d} |{bar}| {total:5,}")

    chart.extend(chart_lines)
    chart.append(f"       {'0':<{width // 2}}{max_tokens:>{width // 2},}")
    chart.append("")

    return "\n".join(chart)


def generate_vertical_chart(conversation_data, height=20, width=50):
    """Generate vertical ASCII chart with context-aware tokens and log scale."""
    token_messages, context_aware_tokens, max_tokens = calculate_cumulative_tokens(
        conversation_data
    )
    if not token_messages:
        return ""

    # Take last 20 data points for the vertical chart (more detailed view)
    display_count = min(20, len(context_aware_tokens))
    data_points = context_aware_tokens[-display_count:]
    display_messages = token_messages[-display_count:]

    # Create spaced columns: each message gets 2 chars (1 for bar + 1 for space)
    chart_width = len(data_points) * 2 - 1  # Remove trailing space

    # Fibonacci-like logarithmic scale
    scale_points = [
        1,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
        10000,
        25000,
        50000,
        100000,
        200000,
    ]

    chart = []
    chart.append("TOKEN PROGRESSION CHART (Vertical)")
    chart.append("-" * 38)
    chart.append("Context-Aware Range: 1 → 200,000 tokens (log scale)")
    chart.append("")

    # Create vertical chart grid using logarithmic scale
    chart_lines = []
    # Use the scale points in reverse order for top-to-bottom display
    display_points = (
        scale_points[-height:] if len(scale_points) > height else scale_points
    )
    display_points = list(reversed(display_points))

    for threshold in display_points:
        # Format the threshold value
        if threshold >= 100000:
            label = f"{int(threshold / 1000)}K"
        elif threshold >= 1000:
            if threshold % 1000 == 0:
                label = f"{int(threshold / 1000)}K"
            else:
                label = f"{threshold / 1000:.1f}K"
        else:
            label = str(threshold)

        line_parts = [f"{label:>6} |"]

        # Create spaced columns for each data point
        for i in range(len(data_points)):
            if data_points[i] >= threshold:
                line_parts.append("█")
            else:
                line_parts.append(" ")

            # # Add space between columns (except for last column)
            # if i < len(data_points) - 1:
            #     line_parts.append(" ")

        line_parts.append("|")
        chart_lines.append("".join(line_parts))

    chart.extend(chart_lines)

    # Add bottom line with message numbers
    chart.append("     0 " + "─" * chart_width)

    # Add message number labels aligned with spaced columns
    label_line = (
        "       "  # Start with Y-axis spacing (7 spaces to match Y-axis labels)
    )

    # Position labels under each column
    for i, msg_data in enumerate(display_messages):
        msg_num = str(msg_data["message_index"])

        # Add the message number
        label_line += msg_num

        # Add spacing to align with next column (except for last)
        if i < len(display_messages) - 1:
            # Calculate spaces needed to reach next column position
            # Each column is 2 chars wide (1 for bar + 1 for space)
            spaces_needed = 2 - len(msg_num)
            if spaces_needed < 1:
                spaces_needed = 1  # Minimum 1 space
            label_line += " " * spaces_needed

    chart.append(label_line)
    chart.append("")

    return "\n".join(chart)


def generate_velocity_chart(conversation_data, height=15, width=50):
    """Generate ASCII line chart showing token consumption velocity."""
    if not conversation_data:
        return ""

    # Get messages with tokens and calculate per-message consumption
    token_messages = [
        msg
        for msg in conversation_data
        if msg["input_tokens"] > 0 or msg["output_tokens"] > 0
    ]
    if len(token_messages) < 5:
        return ""

    # Calculate token delta between messages (velocity)
    velocities = []
    for i in range(1, len(token_messages)):
        (token_messages[i - 1]["input_tokens"] + token_messages[i - 1]["output_tokens"])
        curr_total = (
            token_messages[i]["input_tokens"] + token_messages[i]["output_tokens"]
        )
        velocities.append(curr_total)

    if not velocities:
        return ""

    # Take last 10 data points for the chart (to match other charts)
    data_points = velocities[-10:] if len(velocities) > 10 else velocities
    max_velocity = max(data_points) if data_points else 1
    min_velocity = min(data_points) if data_points else 0

    if max_velocity == min_velocity:
        return ""

    chart = []
    chart.append("TOKEN VELOCITY CHART (per message)")
    chart.append("-" * 35)
    chart.append(f"Range: {min_velocity} - {max_velocity:,} tokens/msg")
    chart.append("")

    # Create line chart grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Plot the line
    for i, velocity in enumerate(data_points):
        if i >= width:
            break

        # Normalize velocity to chart height
        normalized = int(
            ((velocity - min_velocity) / (max_velocity - min_velocity)) * (height - 1)
        )
        y = height - 1 - normalized  # Invert Y axis

        # Draw the point
        if 0 <= y < height:
            grid[y][i] = "●"

            # Draw connecting lines to next point
            if i < len(data_points) - 1 and i + 1 < width:
                next_velocity = data_points[i + 1]
                next_normalized = int(
                    ((next_velocity - min_velocity) / (max_velocity - min_velocity))
                    * (height - 1)
                )
                next_y = height - 1 - next_normalized

                # Draw line between points
                if y != next_y:
                    step = 1 if next_y > y else -1
                    for line_y in range(y + step, next_y, step):
                        if 0 <= line_y < height:
                            grid[line_y][i] = "│"
                elif i + 1 < width:
                    grid[y][i + 1] = "●"

    # Add Y-axis labels and render grid
    for i, row in enumerate(grid):
        y_value = int(
            min_velocity
            + ((height - 1 - i) / (height - 1)) * (max_velocity - min_velocity)
        )
        "".join(row)
        chart.append(f"{y_value:4d} |{''.join(row)}")

    # Add X-axis
    chart.append("     " + "─" * width)
    chart.append(f"     {'1':<10}{'Messages':<15}{len(data_points):>10}")
    chart.append("")

    return "\n".join(chart)


def generate_template_variables(token_data):
    """Generate all template variables for the report."""
    if not token_data:
        return None

    # Calculate derived values
    other_messages = (
        token_data["message_count"]
        - token_data["user_messages"]
        - token_data["assistant_messages"]
    )

    # Format token values with commas
    cumulative_input_tokens = f"{token_data['total_input_tokens']:,}"
    cumulative_output_tokens = f"{token_data['total_output_tokens']:,}"
    input_tokens = token_data["total_input_tokens"]
    output_tokens = token_data["total_output_tokens"]
    total_tokens_val = token_data["total_tokens"]
    total_io_tokens = f"{total_tokens_val:,} ({input_tokens:,} + {output_tokens:,})"

    latest_cache_creation_tokens = f"{token_data['latest_cache_creation_tokens']:,}"
    latest_cache_read_tokens = f"{token_data['latest_cache_read_tokens']:,}"

    # Cache efficiency section
    cache_efficiency_section = ""
    if token_data["total_input_tokens"] > 0:
        cache_read = token_data["total_cache_read_tokens"]
        input_tokens_val = token_data["total_input_tokens"]
        total_input_context = input_tokens_val + cache_read
        cache_efficiency = (cache_read / total_input_context) * 100

        cache_efficiency_section = (
            f"Cache Efficiency:               {cache_efficiency:.1f}% "
            f"({cache_read:,} / {total_input_context:,})"
        )

    # Generate charts
    horizontal_chart = generate_ascii_chart(token_data["conversation_data"])
    vertical_chart = generate_vertical_chart(token_data["conversation_data"])

    # Generate context window components
    context_window_labels, context_window_meter, current_segment = generate_context_window_section(
        token_data
    )
    
    # Calculate context-aware tokens for display
    context_aware_tokens, _ = calculate_context_aware_latest_tokens(token_data["conversation_data"])

    return {
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversation_file": Path(token_data["file_path"]).name,
        "processing_time": f"{token_data.get('generation_time', 0):.3f}s",
        "io_token_count": f"{context_aware_tokens:,}",
        "context_window_labels": context_window_labels,
        "context_window_meter": context_window_meter,
        "current_segment": current_segment,
        "total_messages": token_data["message_count"],
        "user_messages": token_data["user_messages"],
        "assistant_messages": token_data["assistant_messages"],
        "other_messages": other_messages,
        "cumulative_input_tokens": cumulative_input_tokens,
        "cumulative_output_tokens": cumulative_output_tokens,
        "total_io_tokens": total_io_tokens,
        "latest_cache_creation_tokens": latest_cache_creation_tokens,
        "latest_cache_read_tokens": latest_cache_read_tokens,
        "cache_efficiency_section": cache_efficiency_section,
        "horizontal_chart": horizontal_chart if horizontal_chart else "",
        "vertical_chart": vertical_chart if vertical_chart else "",
    }


def calculate_context_aware_tokens(conversation_data):
    """Central function to calculate context-aware tokens for all messages.
    
    Returns:
        tuple: (context_aware_data, latest_tokens, latest_segment)
        - context_aware_data: List of dicts with message_index and context_aware_tokens
        - latest_tokens: Context-aware tokens for the latest message
        - latest_segment: Current segment number (1-based)
    """
    if not conversation_data:
        return [], 0, 0
    
    context_aware_data = []
    latest_context_aware = 0
    latest_segment = 0
    
    # Group by compacting segments and calculate context-aware tokens
    segments = {}
    for msg in conversation_data:
        segment = msg["compacting_segment"]
        if segment not in segments:
            segments[segment] = []
        segments[segment].append(msg)
        latest_segment = max(latest_segment, segment)
    
    # Calculate context-aware tokens for each segment
    for segment in sorted(segments.keys()):
        segment_msgs = segments[segment]
        cumulative_io = 0
        
        for msg in segment_msgs:
            cumulative_io += msg["input_tokens"] + msg["output_tokens"]
            context_aware_value = cumulative_io + msg["cache_read"]
            
            context_aware_data.append({
                "message_index": msg["message_index"],
                "context_aware_tokens": context_aware_value,
                "compacting_segment": segment
            })
            
            # Track the latest value
            latest_context_aware = context_aware_value
    
    return context_aware_data, latest_context_aware, latest_segment + 1  # +1 for 1-based numbering


def calculate_context_aware_latest_tokens(conversation_data):
    """Calculate latest segment's context-aware tokens and segment info."""
    _, latest_tokens, latest_segment = calculate_context_aware_tokens(conversation_data)
    return latest_tokens, latest_segment


def generate_context_window_section(token_data):
    """Generate the context window visualization components using context-aware tokens."""
    # Calculate context-aware token usage for the latest segment
    context_aware_tokens, current_segment = calculate_context_aware_latest_tokens(token_data["conversation_data"])
    context_limit = 200000
    display_markers = [1, 25, 100, 500, 2500, 10000, 50000, 200000]
    segment_width = 5  # 5 characters per segment

    # Create marker labels line with proper spacing
    marker_line = ""
    for i, marker in enumerate(display_markers):
        if marker >= 100000:
            label = f"{int(marker / 1000)}K"
        elif marker >= 1000:
            if marker % 1000 == 0:
                label = f"{int(marker / 1000)}K"
            else:
                label = f"{marker / 1000:.1f}K"
        else:
            label = str(marker)

        if i == 0:
            marker_line += label
        else:
            # Calculate spacing to align with segment boundaries
            # Each segment is 5 chars + 1 separator, so total is 6 chars per segment
            # Position label at the start of each segment
            current_pos = len(marker_line)
            target_pos = i * 6  # 6 chars per segment (5 + 1 separator)
            spaces_needed = target_pos - current_pos
            marker_line += " " * max(1, spaces_needed) + label

    # Create the meter bar
    meter_line = "|"
    if context_aware_tokens > 0:
        # Calculate which segments to fill based on context-aware token count
        for i in range(len(display_markers) - 1):
            segment_min = display_markers[i]
            segment_max = display_markers[i + 1]

            # Determine fill level for this segment
            if context_aware_tokens >= segment_max:
                # Completely fill this segment
                filled_chars = segment_width
            elif context_aware_tokens >= segment_min:
                # Partially fill this segment
                segment_progress = (context_aware_tokens - segment_min) / (
                    segment_max - segment_min
                )
                filled_chars = int(segment_progress * segment_width)
                # Ensure at least 1 character is filled if we're in this range
                if filled_chars == 0 and context_aware_tokens > segment_min:
                    filled_chars = 1
            else:
                # Don't fill this segment at all
                filled_chars = 0

            # Add the segment visualization
            meter_line += "█" * filled_chars + "░" * (segment_width - filled_chars)

            # Add separator between segments (except after the last segment)
            if i < len(display_markers) - 2:
                meter_line += "|"

        # Add position marker based on overall progress
        usage_percent = (context_aware_tokens / context_limit) * 100
        if usage_percent < 2:
            meter_line += "| ◀"  # Very low usage
        elif usage_percent < 25:
            meter_line += "| ●"  # Moderate usage
        else:
            meter_line += "| ▶"  # High usage
    else:
        # No tokens - show empty meter
        total_segments = len(display_markers) - 1
        total_width = total_segments * segment_width + (
            total_segments - 1
        )  # segments + separators
        meter_line += "░" * total_width + "|"

    return marker_line, meter_line, current_segment


def generate_token_report_txt(token_data, conversation_path=None):
    """Generate a formatted token consumption report using template."""
    if not token_data:
        # Handle error cases with simplified template
        if conversation_path is None:
            error_msg = (
                "ERROR: No conversation path provided.\n"
                "Please provide the conversation path as a command line argument.\n\n"
                "Usage: python claude_token_report.py <conversation_path>"
            )
        elif not Path(conversation_path).exists():
            error_msg = (
                f"ERROR: Conversation path does not exist: {conversation_path}\n"
                "Please check that the provided path points to a valid directory."
            )
        else:
            error_msg = (
                f"No conversation files found in: {conversation_path}\n"
                "This directory exists but contains no .jsonl conversation files."
            )

        error_template = """============================================================
CLAUDE CONVERSATION TOKEN REPORT
============================================================
Generated: {{ generation_time }}

{{ error_message }}

============================================================"""

        return error_template.replace(
            "{{ generation_time }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ).replace("{{ error_message }}", error_msg)

    # Generate template variables
    template_vars = generate_template_variables(token_data)
    if not template_vars:
        return "Error generating template variables"

    # Replace template variables
    report = REPORT_TEMPLATE
    for key, value in template_vars.items():
        report = report.replace("{{ " + key + " }}", str(value))

    return report


def main():
    """Main function to generate token consumption report."""
    parser = argparse.ArgumentParser(
        description="Generate Claude conversation token consumption report"
    )
    parser.add_argument(
        "conversation_path", help="Path to the Claude conversation directory"
    )

    # Parse command line arguments
    args = parser.parse_args()

    start_time = time.time()

    try:
        # Try to read JSON input from stdin (for Claude hooks)
        # If no input is available, continue anyway
        if not sys.stdin.isatty():
            json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        # Continue if no JSON input or invalid JSON
        # - this is fine for command line usage
        pass

    # Find and analyze the current conversation
    conversation_file = find_current_conversation_file(args.conversation_path)
    token_data = parse_conversation_tokens(conversation_file)

    # Add timing information to the data
    if token_data:
        token_data["generation_time"] = time.time() - start_time

    # Generate and save TXT report
    txt_report = generate_token_report_txt(token_data, args.conversation_path)

    try:
        # Write TXT version using global configuration
        report_file_path = Path(REPORT_PATH) / REPORT_FILENAME
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(txt_report)

        total_time = time.time() - start_time
        print(
            f"Token report generated: {report_file_path} ({total_time:.3f}s)",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"Error writing reports: {e}", file=sys.stderr)

    # Exit code 0 allows the tool call to proceed normally
    sys.exit(0)


if __name__ == "__main__":
    main()
