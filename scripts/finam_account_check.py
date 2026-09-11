"""Verify the live execution path end-to-end, read-only: auth -> account -> orders.

Confirms the three things that must work before Финам Арена starts, without
placing, modifying or cancelling anything:

1. the token actually grants **account** access (the data token might be
   read-only for quotes and nothing else),
2. `FinamBroker.account()` parses the real account payload — equity, cash and
   open positions — rather than only the mocked shape the unit tests use,
3. `FinamBroker.orders()` parses the real order list.

The account id is read from `FINAM_ACCOUNT_ID` (root `.env`); if it is absent,
the ids embedded in the session-details response are printed so it can be
filled in. Neither the secret nor the JWT is ever printed.

This script makes real network calls to api.finam.ru and needs real
credentials; it is not run in CI or in the test suite (see
tests/test_finam_broker.py for the mocked-HTTP unit tests).

**It never writes.** Nothing here places or cancels an order.

Run with: uv run python scripts/finam_account_check.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from moex_backtest.data import FinamAPIError, FinamClient
from moex_backtest.execution import FinamBroker


def main() -> None:
    load_dotenv()
    secret = os.environ.get("FINAM_SECRET_TOKEN")
    if not secret:
        print("FINAM_SECRET_TOKEN is not set (expected in a root .env) — aborting.")
        sys.exit(1)

    account_id = os.environ.get("FINAM_ACCOUNT_ID", "")

    with FinamClient(secret=secret) as client:
        try:
            client.authenticate()
            print("Auth OK (JWT obtained, token itself not printed).")
            details = client._request(
                "POST", "/v1/sessions/details", json={"token": client._ensure_token()}
            )
        except FinamAPIError as exc:
            print(f"Finam API error during auth: {exc}")
            sys.exit(1)

        known_ids = details.get("account_ids") or []
        print(f"Account ids on this token: {known_ids or '(none reported)'}")
        print(f"Token readonly flag:       {details.get('readonly')}")

        if not account_id:
            if not known_ids:
                print(
                    "\nFINAM_ACCOUNT_ID is not set and the token reports no account ids — "
                    "set FINAM_ACCOUNT_ID in .env to continue."
                )
                sys.exit(1)
            account_id = str(known_ids[0])
            print(f"FINAM_ACCOUNT_ID not set; using the first reported id: {account_id}")

        broker = FinamBroker(client, account_id)
        try:
            account = broker.account()
            orders = broker.orders()
        except FinamAPIError as exc:
            print(f"\nFinam API error while reading the account: {exc}")
            print(
                "If this is a 403/404, the token most likely lacks trading scope for this "
                "account — that is the thing to fix before the contest starts."
            )
            sys.exit(1)

    print()
    print(f"Account:            {account.account_id}")
    print(f"Equity:             {account.equity:,.2f}")
    print(f"Unrealized profit:  {account.unrealized_profit:,.2f}")
    for balance in account.cash:
        print(f"Cash:               {balance.amount:,.2f} {balance.currency}")
    print(f"Open positions:     {len(account.positions)}")
    for position in account.positions:
        print(
            f"  {position.symbol:<16} qty={position.quantity:>12,.4f} "
            f"avg={position.average_price:>12,.4f} pnl={position.unrealized_pnl:>12,.2f}"
        )
    print(f"Orders reported:    {len(orders)}")
    working = [order for order in orders if not order.is_terminal]
    print(f"  still working:    {len(working)}")
    for order in working:
        print(
            f"  {order.order_id:<12} {order.symbol:<16} {order.side:<10} "
            f"{order.status} exec={order.executed_quantity:g}/{order.quantity:g}"
        )

    print()
    print("RESULT: auth, account sync and order listing all work against the live API.")
    print("Order placement is exercised only by the mocked unit tests — by design.")


if __name__ == "__main__":
    main()
