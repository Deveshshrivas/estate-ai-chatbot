from unittest.mock import AsyncMock

import pytest

from app.customer_memory import get_customer_profile


class AsyncDatabaseLike:
    """Matches Motor/PyMongo's refusal to be evaluated as a boolean."""

    def __init__(self) -> None:
        self.customer_profiles = type(
            "Collection", (), {"find_one": AsyncMock(return_value={"name": "Devesh"})}
        )()

    def __bool__(self) -> bool:
        raise NotImplementedError("AsyncDatabase has no truth value")


@pytest.mark.asyncio
async def test_explicit_database_is_not_boolean_evaluated() -> None:
    database = AsyncDatabaseLike()

    profile = await get_customer_profile("919999999999", database)

    assert profile == {"name": "Devesh"}
    database.customer_profiles.find_one.assert_awaited_once_with(
        {"phone": "919999999999"}
    )
