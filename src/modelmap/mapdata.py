"""Decide what each record becomes on the map.

Every record ends up in exactly one place:

- a tile of its own, when it has a size;
- a note on another record's tile, when all its files were charged to that
  record (a second name for the same weights);
- the list of records not drawn, when it has no size and no record to
  attach to (weights that were never downloaded, for example).

Nothing is dropped silently: the three counts add up to the number of
records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelmap.records import ModelRecord, Scan
from modelmap.treemap import Item


@dataclass
class MapData:
    """What to draw.

    :param records: The scan's records; items refer to them by index.
    :param items: One per record that gets a tile; ``key`` is the index.
    :param attached: For a tile's record index, the indexes of the records
        noted on it.
    :param not_drawn: Indexes of records with nothing to draw.
    :param total: Sum of the sizes of the items.
    """

    records: list[ModelRecord]
    items: list[Item] = field(default_factory=list)
    attached: dict[int, list[int]] = field(default_factory=dict)
    not_drawn: list[int] = field(default_factory=list)
    total: int = 0


def build(scan: Scan, *, use_disk: bool = False) -> MapData:
    """Sort a scan's records into tiles, notes and records not drawn.

    :param scan: The scan to draw.
    :param use_disk: Size tiles by bytes allocated on disk, not by apparent
        size.
    """
    data = MapData(records=scan.records)
    owners: dict[str, list[int]] = {}
    for index, record in enumerate(scan.records):
        owners.setdefault(f"{record.provider}:{record.name}", []).append(index)

    for index, record in enumerate(scan.records):
        size = record.disk_bytes if use_disk else record.size_bytes
        if size > 0:
            data.items.append(Item(str(index), record.provider, size, record.name))
            data.total += size
            continue
        # Owner names are not unique, and the owner may not be in this scan
        # at all. Attach only when exactly one record answers to the name.
        candidates = {i for owner in record.shared_with for i in owners.get(owner, ())}
        candidates.discard(index)
        if len(candidates) == 1:
            data.attached.setdefault(candidates.pop(), []).append(index)
        else:
            data.not_drawn.append(index)

    # A note belongs on a tile; if its owner has no tile either, it is not drawn.
    drawn = {int(item.key) for item in data.items}
    for owner in [o for o in data.attached if o not in drawn]:
        data.not_drawn += data.attached.pop(owner)
    data.not_drawn.sort()
    return data
