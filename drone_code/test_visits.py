import json
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright

URL = "https://made-to-give.lovable.app/"
TZ = ZoneInfo("America/Los_Angeles")
DEADLINE = datetime(2026, 10, 22, 23, 59, tzinfo=TZ)

COUNTS = {
    "nothing": 42,
    "browse": 15,
    "learn_more": 12,
}

VIEW_TIME_MS = 3000
PLAN_FILE = Path(__file__).with_name("visit_schedule.json")


def save_plan(plan):
    temporary = PLAN_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )
    temporary.replace(PLAN_FILE)


def randomize_pending_times(plan):
    pending = [visit for visit in plan if visit["status"] == "pending"]

    if not pending:
        return

    start = time.time()
    end = DEADLINE.timestamp()

    if start >= end:
        raise ValueError("The deadline has already passed.")

    times = sorted(
        random.uniform(start, end)
        for _ in pending
    )

    for visit, scheduled in zip(pending, times):
        visit["time"] = scheduled


def load_or_create_plan():
    if PLAN_FILE.exists():
        plan = json.loads(
            PLAN_FILE.read_text(encoding="utf-8")
        )

        # Update pending visits if the deadline changed.
        deadline_changed = any(
            visit.get("deadline") != DEADLINE.isoformat()
            for visit in plan
        )

        if deadline_changed:
            randomize_pending_times(plan)

            for visit in plan:
                visit["deadline"] = DEADLINE.isoformat()

            save_plan(plan)

        return plan

    actions = [
        action
        for action, count in COUNTS.items()
        for _ in range(count)
    ]
    random.shuffle(actions)

    plan = [
        {
            "time": None,
            "action": action,
            "status": "pending",
            "deadline": DEADLINE.isoformat(),
        }
        for action in actions
    ]

    randomize_pending_times(plan)
    save_plan(plan)
    return plan


def perform_visit(playwright, action):
    browser = playwright.chromium.launch(headless=True)

    try:
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(30_000)

        page.goto(URL, wait_until="load")
        page.wait_for_timeout(VIEW_TIME_MS)

        if action == "browse":
            page.get_by_role(
                "link",
                name="Browse one-off cake sessions",
                exact=True,
            ).click()

            page.wait_for_url("**/sessions?cta=primary")

        elif action == "learn_more":
            page.get_by_role(
                "link",
                name="Learn more",
                exact=True,
            ).click()

            page.wait_for_url("**/sessions?cta=secondary")

        page.wait_for_timeout(VIEW_TIME_MS)

    finally:
        browser.close()


def main():
    plan = load_or_create_plan()

    print(f"Schedule: {len(plan)} sessions", flush=True)
    print(
        f"Deadline: {DEADLINE:%A, %Y-%m-%d %I:%M %p %Z}",
        flush=True,
    )

    with sync_playwright() as p:
        for number, visit in enumerate(plan, start=1):
            if visit["status"] != "pending":
                continue

            if time.time() >= DEADLINE.timestamp():
                print("Deadline reached. Stopping.", flush=True)
                break

            scheduled = datetime.fromtimestamp(visit["time"], TZ)

            print(
                f"Next: {number}/{len(plan)}, "
                f"{scheduled:%Y-%m-%d %H:%M:%S %Z}, "
                f"action={visit['action']}",
                flush=True,
            )

            while True:
                remaining = visit["time"] - time.time()

                if remaining <= 0:
                    break

                time.sleep(min(30, remaining))

            if time.time() >= DEADLINE.timestamp():
                print("Deadline reached. Stopping.", flush=True)
                break

            # Avoid automatically repeating a partially completed
            # visit after an interruption.
            visit["status"] = "started"
            save_plan(plan)

            try:
                perform_visit(p, visit["action"])
                visit["status"] = "completed"

                print(
                    f"Session {number}: completed",
                    flush=True,
                )

            except Exception as error:
                visit["status"] = "failed"
                visit["error"] = str(error)

                print(
                    f"Session {number}: failed: {error}",
                    flush=True,
                )

            save_plan(plan)

    completed = sum(
        visit["status"] == "completed"
        for visit in plan
    )

    print(
        f"Completed: {completed}/{len(plan)}",
        flush=True,
    )


if __name__ == "__main__":
    main()