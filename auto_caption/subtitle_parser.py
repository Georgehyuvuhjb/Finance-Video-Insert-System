#!/usr/bin/env python3
"""
Subtitle Parser Module
=====================

Parses timestamp script files and converts them to subtitle data structures.
Supports the format: 00:00.05_00:10.68_text_content

This module is shared between OpenCV and FFmpeg caption generators.
"""

import re
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class SubtitleSegment:
    """Data structure for a single subtitle segment"""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    text: str         # Subtitle text content
    
    def duration(self) -> float:
        """Get the duration of this subtitle segment"""
        return self.end_time - self.start_time

class SubtitleParser:
    """Parser for timestamp script files"""
    
    def __init__(self):
        self.segments: List[SubtitleSegment] = []
        
    def parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse timestamp string to seconds
        
        Args:
            timestamp_str: Timestamp in format MM:SS.ms or HH:MM:SS.ms
            
        Returns:
            Time in seconds as float
        """
        # Handle both MM:SS.ms and HH:MM:SS.ms formats
        if timestamp_str.count(':') == 1:
            # MM:SS.ms format
            minutes, seconds = timestamp_str.split(':')
            return float(minutes) * 60 + float(seconds)
        elif timestamp_str.count(':') == 2:
            # HH:MM:SS.ms format
            hours, minutes, seconds = timestamp_str.split(':')
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    
    def parse_script_file(self, script_path: str) -> List[SubtitleSegment]:
        """
        Parse script file with timestamp format
        
        Args:
            script_path: Path to script file
            
        Returns:
            List of SubtitleSegment objects
        """
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script file not found: {script_path}")
        
        segments = []
        
        with open(script_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: 00:00.05_00:10.68_text_content
                parts = line.split('_', 2)
                if len(parts) < 3:
                    print(f"Warning: Skipping invalid line {line_num}: {line}")
                    continue
                
                try:
                    start_time = self.parse_timestamp(parts[0])
                    end_time = self.parse_timestamp(parts[1])
                    text = parts[2]
                    
                    # Validate time order
                    if end_time <= start_time:
                        print(f"Warning: Invalid time range at line {line_num}: {line}")
                        continue
                    
                    segment = SubtitleSegment(
                        start_time=start_time,
                        end_time=end_time,
                        text=text
                    )
                    segments.append(segment)
                    
                except ValueError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        self.segments = segments
        print(f"Successfully parsed {len(segments)} subtitle segments")
        return segments
    
    def get_segments_at_time(self, time_seconds: float) -> List[SubtitleSegment]:
        """
        Get all subtitle segments that should be displayed at a given time
        
        Args:
            time_seconds: Time in seconds
            
        Returns:
            List of active subtitle segments
        """
        active_segments = []
        for segment in self.segments:
            if segment.start_time <= time_seconds <= segment.end_time:
                active_segments.append(segment)
        return active_segments
    
    def export_to_srt(self, output_path: str) -> None:
        """
        Export segments to SRT subtitle format
        
        Args:
            output_path: Path for output SRT file
        """
        def format_srt_time(seconds: float) -> str:
            """Format seconds to SRT time format (HH:MM:SS,mmm)"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_srt_time(segment.start_time)} --> {format_srt_time(segment.end_time)}\n")
                f.write(f"{segment.text}\n\n")
        
        print(f"SRT file exported to: {output_path}")
    
    def get_subtitle_stats(self) -> Dict:
        """
        Get statistics about the parsed subtitles
        
        Returns:
            Dictionary with subtitle statistics
        """
        if not self.segments:
            return {}
        
        total_duration = sum(seg.duration() for seg in self.segments)
        max_text_length = max(len(seg.text) for seg in self.segments)
        min_duration = min(seg.duration() for seg in self.segments)
        max_duration = max(seg.duration() for seg in self.segments)
        
        return {
            'total_segments': len(self.segments),
            'total_subtitle_duration': total_duration,
            'average_segment_duration': total_duration / len(self.segments),
            'min_segment_duration': min_duration,
            'max_segment_duration': max_duration,
            'max_text_length': max_text_length,
            'start_time': self.segments[0].start_time,
            'end_time': self.segments[-1].end_time
        }

def main():
    """Test the subtitle parser"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python subtitle_parser.py <script_file>")
        return
    
    script_file = sys.argv[1]
    parser = SubtitleParser()
    
    try:
        segments = parser.parse_script_file(script_file)
        stats = parser.get_subtitle_stats()
        
        print("\nSubtitle Statistics:")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Total duration: {stats['total_subtitle_duration']:.2f} seconds")
        print(f"Average segment duration: {stats['average_segment_duration']:.2f} seconds")
        print(f"Time range: {stats['start_time']:.2f}s - {stats['end_time']:.2f}s")
        print(f"Max text length: {stats['max_text_length']} characters")
        
        # Export to SRT for testing
        srt_path = script_file.replace('.txt', '.srt')
        parser.export_to_srt(srt_path)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
