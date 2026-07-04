import unittest
from datetime import datetime

from src.preprocessing.preprocessor import (
    normalize_log_line,
    parse_timestamp,
    extract_windows,
)


class TestNormalizeLogLine(unittest.TestCase):

    def test_ip_address(self):
        line = "Client connected from 192.168.1.10"
        self.assertIn("<IP_ADDR>", normalize_log_line(line))

    def test_uuid(self):
        line = "UUID: 550e8400-e29b-41d4-a716-446655440000"
        self.assertIn("<UUID>", normalize_log_line(line))

    def test_timestamp(self):
        line = "2025-06-10 10:15:22 ERROR Database failed"
        self.assertIn("<TIMESTAMP>", normalize_log_line(line))

    def test_user_id(self):
        line = "user_12345 logged in"
        self.assertIn("<USER_ID>", normalize_log_line(line))

    def test_file_path(self):
        line = "Cannot open /var/log/system.log"
        self.assertIn("<FILE_PATH>", normalize_log_line(line))

    def test_memory_address(self):
        line = "Pointer = 0x7ffeefbff618"
        self.assertIn("<MEM_ADDR>", normalize_log_line(line))

    def test_port(self):
        line = "Connected to 10.0.0.1:8080"
        normalized = normalize_log_line(line)
        self.assertIn("<PORT>", normalized)


class TestParseTimestamp(unittest.TestCase):

    def test_valid_space_timestamp(self):
        ts = parse_timestamp("2025-06-10 10:15:22 INFO Started")
        self.assertEqual(
            ts,
            datetime(2025, 6, 10, 10, 15, 22)
        )

    def test_valid_iso_timestamp(self):
        ts = parse_timestamp("2025-06-10T10:15:22 INFO Started")
        self.assertEqual(
            ts,
            datetime(2025, 6, 10, 10, 15, 22)
        )

    def test_invalid_timestamp(self):
        self.assertIsNone(
            parse_timestamp("INFO Server started")
        )


class TestExtractWindows(unittest.TestCase):

    def test_single_window(self):

        logs = [
            "2025-06-10 10:00:00 INFO Start",
            "2025-06-10 10:00:30 INFO Continue",
            "2025-06-10 10:00:45 INFO End",
        ]

        windows = extract_windows(logs)

        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0]), 3)

    def test_gap_creates_new_window(self):

        logs = [
            "2025-06-10 10:00:00 INFO Start",
            "2025-06-10 10:02:30 ERROR Failure",
        ]

        windows = extract_windows(logs)

        self.assertEqual(len(windows), 2)

    def test_max_lines(self):

        logs = [
            f"2025-06-10 10:00:{i:02d} INFO Message {i}"
            for i in range(25)
        ]

        windows = extract_windows(logs)

        self.assertEqual(len(windows), 2)
        self.assertEqual(len(windows[0]), 20)
        self.assertEqual(len(windows[1]), 5)

    def test_empty_input(self):

        windows = extract_windows([])

        self.assertEqual(windows, [])


if __name__ == "__main__":
    unittest.main()