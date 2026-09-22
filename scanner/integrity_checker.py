NeuroFence - Model Integrity Checker

This module verifies the integrity of an LLM model or model file.

It uses SHA-256 hashing to create a fingerprint of the model files.
If a previously trusted hash is available, the current hash can be
compared against it.

IMPORTANT:
An integrity mismatch means that the file is different from the
trusted/reference version. It does NOT automatically mean that the
model is malicious.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import math


class IntegrityChecker:
    """
    Check the integrity of model files using SHA-256 hashes.

    Supported checks:
        - File existence
        - File size
        - SHA-256 hash
        - Hash comparison with a trusted/reference hash
        - Directory-level integrity
    """

    def __init__(
        self,
        algorithm: str = "sha256",
        chunk_size: int = 1024 * 1024,
    ) -> None:
        """
        Initialize the integrity checker.

        Args:
            algorithm:
                Hashing algorithm. Default is SHA-256.

            chunk_size:
                Number of bytes read at a time.
                Default is 1 MB.
        """

        algorithm = algorithm.lower().strip()

        if algorithm not in hashlib.algorithms_available:
            raise ValueError(
                f"Unsupported hashing algorithm: {algorithm}"
            )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        self.algorithm = algorithm
        self.chunk_size = int(chunk_size)

    def _create_hash(self):
        """
        Create a new hash object.
        """

        return hashlib.new(self.algorithm)

    def calculate_file_hash(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Calculate the cryptographic hash of a file.

        Args:
            file_path:
                Path to the file.

        Returns:
            Hexadecimal hash string.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        hash_object = self._create_hash()

        with path.open("rb") as file:

            while True:

                chunk = file.read(
                    self.chunk_size
                )

                if not chunk:
                    break

                hash_object.update(chunk)

        return hash_object.hexdigest()

    def get_file_metadata(
        self,
        file_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Get basic metadata and hash information for a file.
        """

        path = Path(file_path)

        if not path.exists():
            return {
                "path": str(path),
                "exists": False,
                "is_file": False,
                "size_bytes": 0,
                "hash": None,
                "error": "File does not exist.",
            }

        if not path.is_file():
            return {
                "path": str(path),
                "exists": True,
                "is_file": False,
                "size_bytes": 0,
                "hash": None,
                "error": "Path is not a file.",
            }

        try:

            file_hash = self.calculate_file_hash(path)

            return {
                "path": str(path),
                "exists": True,
                "is_file": True,
                "size_bytes": path.stat().st_size,
                "hash_algorithm": self.algorithm,
                "hash": file_hash,
                "error": None,
            }

        except Exception as exc:

            return {
                "path": str(path),
                "exists": True,
                "is_file": True,
                "size_bytes": path.stat().st_size,
                "hash_algorithm": self.algorithm,
                "hash": None,
                "error": str(exc),
            }

    def compare_hashes(
        self,
        current_hash: str,
        trusted_hash: str,
    ) -> Dict[str, Any]:
        """
        Compare a current hash with a trusted/reference hash.
        """

        current_hash = str(
            current_hash or ""
        ).strip().lower()

        trusted_hash = str(
            trusted_hash or ""
        ).strip().lower()

        if not current_hash:
            return {
                "match": False,
                "status": "unknown",
                "message": "Current hash is missing.",
            }

        if not trusted_hash:
            return {
                "match": False,
                "status": "no_reference",
                "message": (
                    "No trusted reference hash was provided."
                ),
            }

        match = current_hash == trusted_hash

        if match:
            return {
                "match": True,
                "status": "verified",
                "message": (
                    "Current hash matches the trusted hash."
                ),
            }

        return {
            "match": False,
            "status": "mismatch",
            "message": (
                "Current hash does not match the trusted hash."
            ),
        }

    def check_file(
        self,
        file_path: str | Path,
        trusted_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform a complete integrity check on one file.
        """

        metadata = self.get_file_metadata(
            file_path
        )

        if not metadata["exists"]:
            return {
                **metadata,
                "integrity_status": "missing",
                "integrity_verified": False,
            }

        if not metadata["is_file"]:
            return {
                **metadata,
                "integrity_status": "invalid_path",
                "integrity_verified": False,
            }

        if metadata["hash"] is None:
            return {
                **metadata,
                "integrity_status": "hash_error",
                "integrity_verified": False,
            }

        if trusted_hash is None:

            return {
                **metadata,
                "integrity_status": "hash_generated",
                "integrity_verified": None,
                "message": (
                    "Hash generated, but no trusted "
                    "reference hash was supplied."
                ),
            }

        comparison = self.compare_hashes(
            current_hash=metadata["hash"],
            trusted_hash=trusted_hash,
        )

        return {
            **metadata,
            "integrity_status": comparison["status"],
            "integrity_verified": comparison["match"],
            "message": comparison["message"],
            "trusted_hash": trusted_hash,
        }

    def calculate_directory_hash(
        self,
        directory_path: str | Path,
    ) -> str:
        """
        Calculate a deterministic hash representing all files
        inside a directory.

        Files are sorted by their relative path so that the same
        directory contents produce the same result regardless of
        traversal order.
        """

        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Path is not a directory: {directory}"
            )

        files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
        )

        directory_hash = self._create_hash()

        for file_path in files:

            relative_path = file_path.relative_to(
                directory
            )

            path_bytes = str(
                relative_path
            ).replace("\\", "/").encode(
                "utf-8"
            )

            directory_hash.update(
                path_bytes
            )

            file_hash = self.calculate_file_hash(
                file_path
            )

            directory_hash.update(
                file_hash.encode("utf-8")
            )

        return directory_hash.hexdigest()

    def check_directory(
        self,
        directory_path: str | Path,
        trusted_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check the integrity of all files in a model directory.
        """

        directory = Path(directory_path)

        if not directory.exists():

            return {
                "path": str(directory),
                "exists": False,
                "file_count": 0,
                "directory_hash": None,
                "integrity_status": "missing",
                "integrity_verified": False,
            }

        if not directory.is_dir():

            return {
                "path": str(directory),
                "exists": True,
                "file_count": 0,
                "directory_hash": None,
                "integrity_status": "invalid_path",
                "integrity_verified": False,
            }

        files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
        )

        file_results: List[Dict[str, Any]] = []

        for file_path in files:

            file_results.append(
                self.check_file(file_path)
            )

        try:

            directory_hash = (
                self.calculate_directory_hash(
                    directory
                )
            )

        except Exception as exc:

            return {
                "path": str(directory),
                "exists": True,
                "file_count": len(files),
                "directory_hash": None,
                "files": file_results,
                "integrity_status": "hash_error",
                "integrity_verified": False,
                "error": str(exc),
            }

        if trusted_hash is None:

            return {
                "path": str(directory),
                "exists": True,
                "file_count": len(files),
                "directory_hash": directory_hash,
                "hash_algorithm": self.algorithm,
                "files": file_results,
                "integrity_status": "hash_generated",
                "integrity_verified": None,
                "message": (
                    "Directory hash generated, but no trusted "
                    "reference hash was supplied."
                ),
            }

        comparison = self.compare_hashes(
            current_hash=directory_hash,
            trusted_hash=trusted_hash,
        )

        return {
            "path": str(directory),
            "exists": True,
            "file_count": len(files),
            "directory_hash": directory_hash,
            "hash_algorithm": self.algorithm,
            "files": file_results,
            "integrity_status": comparison["status"],
            "integrity_verified": comparison["match"],
            "message": comparison["message"],
            "trusted_hash": trusted_hash,
        }

    def verify_expected_files(
        self,
        directory_path: str | Path,
        expected_files: Iterable[str],
    ) -> Dict[str, Any]:
        """
        Verify that expected model files exist in a directory.
        """

        directory = Path(directory_path)

        expected_files = list(expected_files)

        results = []

        for filename in expected_files:

            file_path = directory / filename

            exists = (
                file_path.exists()
                and file_path.is_file()
            )

            results.append(
                {
                    "file": filename,
                    "exists": exists,
                    "path": str(file_path),
                }
            )

        missing_files = [
            item["file"]
            for item in results
            if not item["exists"]
        ]

        return {
            "directory": str(directory),
            "expected_count": len(expected_files),
            "found_count": (
                len(expected_files)
                - len(missing_files)
            ),
            "missing_count": len(missing_files),
            "missing_files": missing_files,
            "all_present": len(missing_files) == 0,
            "files": results,
        }

    def generate_integrity_summary(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a compact result for the UI/report.
        """

        status = result.get(
            "integrity_status",
            "unknown",
        )

        verified = result.get(
            "integrity_verified"
        )

        if status == "verified":
            risk_level = "low"

        elif status == "mismatch":
            risk_level = "high"

        elif status in {
            "missing",
            "invalid_path",
            "hash_error",
        }:
            risk_level = "unknown"

        else:
            risk_level = "informational"

        return {
            "integrity_status": status,
            "integrity_verified": verified,
            "risk_level": risk_level,
            "hash_algorithm": result.get(
                "hash_algorithm",
                self.algorithm,
            ),
            "message": result.get(
                "message",
                "",
            ),
        }


def calculate_hash(
    file_path: str | Path,
) -> str:
    """
    Convenience function for calculating a SHA-256 hash.
    """

    checker = IntegrityChecker()

    return checker.calculate_file_hash(
        file_path
    )


def check_integrity(
    file_path: str | Path,
    trusted_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for checking file integrity.
    """

    checker = IntegrityChecker()

    return checker.check_file(
        file_path=file_path,
        trusted_hash=trusted_hash,
    )


if __name__ == "__main__":

    print("=" * 65)
    print("NEUROFENCE INTEGRITY CHECKER TEST")
    print("=" * 65)

    # ---------------------------------------------------------
    # Create a small temporary test file.
    # ---------------------------------------------------------

    test_file = Path(
        "neurofence_integrity_test.txt"
    )

    test_file.write_text(
        "NeuroFence integrity test file.",
        encoding="utf-8",
    )

    checker = IntegrityChecker()

    # ---------------------------------------------------------
    # Generate hash.
    # ---------------------------------------------------------

    file_hash = checker.calculate_file_hash(
        test_file
    )

    print("\nTest file:")
    print(test_file)

    print("\nSHA-256:")
    print(file_hash)

    # ---------------------------------------------------------
    # Generate metadata.
    # ---------------------------------------------------------

    metadata = checker.get_file_metadata(
        test_file
    )

    print("\nFile metadata:")
    print(metadata)

    # ---------------------------------------------------------
    # Check without reference.
    # ---------------------------------------------------------

    result = checker.check_file(
        test_file
    )

    print("\nIntegrity check without reference:")
    print(result)

    # ---------------------------------------------------------
    # Check using the generated hash as trusted hash.
    # ---------------------------------------------------------

    verified_result = checker.check_file(
        file_path=test_file,
        trusted_hash=file_hash,
    )

    print("\nIntegrity check with matching hash:")
    print(verified_result)

    # ---------------------------------------------------------
    # Test a deliberately different reference hash.
    # ---------------------------------------------------------

    mismatch_result = checker.check_file(
        file_path=test_file,
        trusted_hash="0" * 64,
    )

    print("\nIntegrity check with mismatching hash:")
    print(mismatch_result)

    # ---------------------------------------------------------
    # Generate UI summary.
    # ---------------------------------------------------------

    summary = checker.generate_integrity_summary(
        verified_result
    )

    print("\nIntegrity summary:")
    print(summary)

    # ---------------------------------------------------------
    # Clean up test file.
    # ---------------------------------------------------------

    try:
        test_file.unlink()
    except OSError:
        pass

    print("\n" + "=" * 65)
    print("INTEGRITY CHECKER TEST COMPLETED")
    print("=" * 65)
