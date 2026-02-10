# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Smoke test fixtures for basic server validation.

Smoke tests are minimal tests that verify the server can start and handle
basic operations. They run quickly and catch critical regressions.
"""


# Smoke tests reuse E2E fixtures from parent conftest
# This file exists to:
# 1. Document smoke test philosophy
# 2. Provide smoke-specific fixtures if needed in the future
# 3. Maintain clear test organization

# Import E2E fixtures for reuse
pytest_plugins = ["tests.e2e.conftest"]
