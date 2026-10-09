
from tree_classification.tracking.file_tracker import (
    FileTracker,
)


def test_file_tracker(tmp_path):
    tracker = FileTracker(tmp_path)
    test_file = tmp_path / "test.laz"
    
    # Not processed initially
    assert not tracker.is_processed(test_file)
    
    # Mark as processed
    tracker.mark_processed(test_file)
    assert tracker.is_processed(test_file)
    
    # Error tracking
    assert len(tracker.get_errors()) == 0
    tracker.log_error(test_file)
    errors = tracker.get_errors()
    assert len(errors) == 1
    assert errors[0] == test_file.resolve()
    
    # Duplicate prevention
    tracker.mark_processed(test_file)
    tracker.log_error(test_file)
    
    assert len(tracker.processed_file.read_text().splitlines()) == 1
        
    assert len(tracker.error_file.read_text().splitlines()) == 1
