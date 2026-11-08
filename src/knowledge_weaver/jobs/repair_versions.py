"""Finite, explicit processor upgrades shared by request/commit/restore checks."""

EXTRACTION_UPGRADES = {
    'prompt:7': ('prompt:3', 'prompt:4', 'prompt:5', 'prompt:6'),
    'prompt:8': ('prompt:3', 'prompt:4', 'prompt:5', 'prompt:6', 'prompt:7'),
    'prompt:9': ('prompt:3', 'prompt:4', 'prompt:5', 'prompt:6', 'prompt:7', 'prompt:8'),
    'prompt:10': ('prompt:3', 'prompt:4', 'prompt:5', 'prompt:6', 'prompt:7', 'prompt:8', 'prompt:9'),
}
MAX_REPAIR_DEPTH = len(EXTRACTION_UPGRADES)


def allowed_upgrade(new_version, old_version):
    return old_version in EXTRACTION_UPGRADES.get(new_version, ())


def upgrade_sql(new_column, old_column):
    # Callers pass internal SQL column expressions, never external input.
    return '(' + ' OR '.join(
        "(" + new_column + "='" + version + "' AND " + old_column + " IN ("
        + ','.join("'" + old + "'" for old in parents) + '))'
        for version, parents in EXTRACTION_UPGRADES.items()
    ) + ')'
