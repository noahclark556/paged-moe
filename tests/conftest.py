# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later

# The training tests are scripts (python tests/test_train_methods.py). They put
# the whole process in training mode on import (slab, verify and the rest off),
# which would leak into the serving tests if pytest collected them.
collect_ignore_glob = ["test_train_*.py"]
