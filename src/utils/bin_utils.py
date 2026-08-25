import os
import re
import tempfile
import glob
from collections import Counter
from datetime import date, timedelta
from typing import Tuple, List, Optional

# Valid IFCB bin PIDs come in two eras, and each implies its own day-directory name:
#
#   new style   D{YYYYMMDD}T{HHMMSS}_IFCB{NNN}   under {YEAR}/D{YYYYMMDD}/
#   old style   IFCB{N}_{YYYY}_{DDD}_{HHMMSS}    under {YEAR}/IFCB{N}_{YYYY}_{DDD}/
#
# Filenames alone cannot distinguish good data from calibration or scratch data --
# a beads acquisition has a perfectly valid PID. What marks it is *where* it sits,
# so validation is on the whole relative path.
NEW_STYLE_PID = re.compile(r"^D(\d{4})(\d{2})(\d{2})T\d{6}_IFCB\d+$")
OLD_STYLE_PID = re.compile(r"^IFCB\d+_(\d{4})_(\d{3})_\d{6}$")

NEW_STYLE_DAY_DIR = re.compile(r"^D(\d{4})(\d{2})(\d{2})$")
OLD_STYLE_DAY_DIR = re.compile(r"^IFCB\d+_(\d{4})_(\d{3})$")

# A day directory collects an acquisition session, which can run past midnight, so
# the bins inside may carry the following day's date. Anything further out than this
# is treated as misfiled rather than as a session boundary.
DEFAULT_DAY_TOLERANCE = 1


def _year_day_to_date(year: str, year_day: str) -> Optional[date]:
    try:
        return date(int(year), 1, 1) + timedelta(days=int(year_day) - 1)
    except ValueError:
        return None


def _ymd_to_date(year: str, month: str, day: str) -> Optional[date]:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def classify_bin_path(relative_path: str, day_tolerance: int = DEFAULT_DAY_TOLERANCE):
    """Classify one .adc path relative to the data directory.

    Returns:
        tuple: (pid, bin_type, reason). On success reason is None and bin_type is
        'D' (new style) or 'I' (old style). On rejection pid may still be set, and
        reason describes why the path was rejected.
    """
    parts = relative_path.split(os.sep)

    if len(parts) != 3:
        return None, None, f"expected {{year}}/{{day}}/{{pid}}.adc, got {len(parts)} path components"

    year_dir, day_dir, filename = parts
    pid = filename[:-4] if filename.endswith(".adc") else filename

    if not re.fullmatch(r"\d{4}", year_dir):
        return pid, None, f"top-level directory {year_dir!r} is not a 4-digit year"

    new_pid = NEW_STYLE_PID.match(pid)
    old_pid = OLD_STYLE_PID.match(pid)
    if new_pid:
        bin_type = "D"
        pid_date = _ymd_to_date(*new_pid.groups())
        day_match = NEW_STYLE_DAY_DIR.match(day_dir)
        day_date = _ymd_to_date(*day_match.groups()) if day_match else None
    elif old_pid:
        bin_type = "I"
        pid_date = _year_day_to_date(*old_pid.groups())
        day_match = OLD_STYLE_DAY_DIR.match(day_dir)
        day_date = _year_day_to_date(*day_match.groups()) if day_match else None
    else:
        return pid, None, "PID matches neither the new nor the old naming convention"

    if pid_date is None:
        return pid, bin_type, "PID encodes an invalid date"
    if day_date is None:
        # Catches beads/temp/skip and anything else that is not a day directory.
        return pid, bin_type, f"directory {day_dir!r} is not a {bin_type}-style day directory"
    if year_dir != f"{pid_date.year:04d}":
        return pid, bin_type, f"year directory {year_dir!r} does not match PID year {pid_date.year}"

    delta = abs((pid_date - day_date).days)
    if delta > day_tolerance:
        return pid, bin_type, f"PID date is {delta} days from day directory {day_dir!r}"

    return pid, bin_type, None


def find_bins_by_type(
    data_dir: str,
    bin_type: str,
    validate_paths: bool = True,
    day_tolerance: int = DEFAULT_DAY_TOLERANCE,
) -> Tuple[List[str], Counter]:
    """Find all bins of the specified type (I or D) in the data directory.

    Args:
        data_dir: Directory containing IFCB point cloud data
        bin_type: Type of bins to find ('I' or 'D')
        validate_paths: Require each .adc to sit at {year}/{day}/{pid}.adc with the
            directories agreeing with the PID's own date. This excludes calibration
            and scratch trees (beads, temp, data_temp, skip) whose files have valid
            PIDs, and excludes nested duplicate trees, which sit at a deeper path.
        day_tolerance: Days a PID's date may differ from its day directory.

    Returns:
        tuple: (pids, rejections) where pids is a de-duplicated list of PIDs and
        rejections counts rejected paths by reason (empty when validate_paths is
        False).
    """
    adc_files = glob.glob(os.path.join(data_dir, "**", "*.adc"), recursive=True)

    pids: List[str] = []
    seen = set()
    rejections: Counter = Counter()

    for adc_file in sorted(adc_files):
        if not validate_paths:
            pid = os.path.splitext(os.path.basename(adc_file))[0]
            # Preserve the historical prefix test when validation is disabled.
            if pid.startswith(bin_type) and pid not in seen:
                seen.add(pid)
                pids.append(pid)
            continue

        pid, path_bin_type, reason = classify_bin_path(
            os.path.relpath(adc_file, data_dir), day_tolerance
        )
        if reason is not None:
            rejections[reason] += 1
            continue
        if path_bin_type != bin_type:
            continue
        # De-duplicate: the same PID can appear at more than one path, and scoring
        # it twice would double-weight that bin in the resulting distribution.
        if pid in seen:
            rejections["duplicate PID already found at another path"] += 1
            continue
        seen.add(pid)
        pids.append(pid)

    return pids, rejections


def create_bin_type_id_file(
    data_dir: str,
    bin_type: str,
    validate_paths: bool = True,
    day_tolerance: int = DEFAULT_DAY_TOLERANCE,
    logger=None,
) -> Tuple[Optional[str], int]:
    """Create a temporary ID file containing only bins of the specified type (I or D).

    Args:
        data_dir: Directory containing IFCB point cloud data
        bin_type: Type of bins to include ('I' or 'D')
        validate_paths: See :func:`find_bins_by_type`.
        day_tolerance: See :func:`find_bins_by_type`.
        logger: Optional logger; rejection counts are reported through it so that
            excluded data is visible in the flow run rather than silently dropped.

    Returns:
        Tuple of (temp_file_path, number_of_bins_found)
    """
    filtered_pids, rejections = find_bins_by_type(
        data_dir, bin_type, validate_paths=validate_paths, day_tolerance=day_tolerance
    )

    if rejections:
        total = sum(rejections.values())
        message = f"{bin_type} bins: excluded {total} .adc path(s) under {data_dir}"
        if logger is not None:
            logger.info(message)
            for reason, count in rejections.most_common():
                logger.info(f"  {count}: {reason}")
        else:
            print(message)
            for reason, count in rejections.most_common():
                print(f"  {count}: {reason}")

    if not filtered_pids:
        return None, 0

    # Create temporary ID file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix=f'{bin_type}_bins_')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            for pid in filtered_pids:
                f.write(f"{pid}\n")
    except:
        os.unlink(temp_path)
        raise

    return temp_path, len(filtered_pids)
