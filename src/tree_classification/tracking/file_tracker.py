from pathlib import Path

class FileTracker:
    """Manages tracking of processed and errored files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.output_dir / "processed.txt"
        self.error_file = self.output_dir / "error_files.txt"
        
        self.processed_file.touch(exist_ok=True)
        self.error_file.touch(exist_ok=True)
        
    def _read_lines(self, file_path: Path) -> set[str]:
        if not file_path.exists():
            return set()
        return {line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            
    def _append_line(self, file_path: Path, line: str) -> None:
        with file_path.open("a", encoding="utf-8") as f:
            f.write(f"{line}\n")

    def is_processed(self, path: Path) -> bool:
        """Check if a file has already been successfully processed."""
        processed = self._read_lines(self.processed_file)
        return str(path.resolve()) in processed

    def mark_processed(self, path: Path) -> None:
        """Mark a file as successfully processed."""
        if not self.is_processed(path):
            self._append_line(self.processed_file, str(path.resolve()))

    def log_error(self, path: Path) -> None:
        """Log a file that encountered an error during processing."""
        errors = self._read_lines(self.error_file)
        if str(path.resolve()) not in errors:
            self._append_line(self.error_file, str(path.resolve()))
            
    def get_errors(self) -> list[Path]:
        """Return a list of all files that encountered errors."""
        return [Path(p) for p in self._read_lines(self.error_file)]
