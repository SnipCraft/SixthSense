
def decide_movement_direction(directional_obstacles):

    left_blocked = bool(directional_obstacles["LEFT"])
    centre_blocked = bool(directional_obstacles["CENTRE"])
    right_blocked = bool(directional_obstacles["RIGHT"])

    if not left_blocked and not centre_blocked and not right_blocked:
        return "NO OBSTACLE DETECTED"

    if left_blocked and right_blocked:
        return "STOP"

    if left_blocked and not centre_blocked:
        return "MOVE RIGHT"

    if right_blocked and not centre_blocked:
        return "MOVE LEFT"

    if centre_blocked:
        if not right_blocked:
            return "MOVE RIGHT"

        if not left_blocked:
            return "MOVE LEFT"

        return "STOP"
        
def stabilize_movement_decision(
    current_decision,
    previous_decision,
    consecutive_count
):

    # STOP must be immediate
    if current_decision == "STOP":
        return "STOP", "STOP", 2

        # No obstacle detected should also be immediate
    if current_decision == "NO OBSTACLE DETECTED":
        return "NO OBSTACLE DETECTED", "NO OBSTACLE DETECTED", 0

    # Count consecutive identical decisions
    if current_decision == previous_decision:
        consecutive_count += 1
    else:
        consecutive_count = 1

    # Confirm after 2 consecutive frames
    if consecutive_count >= 2:
        return current_decision, current_decision, consecutive_count

    return "CHECKING...", current_decision, consecutive_count


test_cases = [
    ("LEFT", [True, False, False]),
    ("RIGHT", [False, False, True]),
    ("CENTRE", [False, True, False]),
    ("LEFT + CENTRE", [True, True, False]),
    ("CENTRE + RIGHT", [False, True, True]),
    ("LEFT + RIGHT", [True, False, True]),
    ("ALL THREE", [True, True, True]),
    ("NO OBSTACLE", [False, False, False])
]

for name, values in test_cases:

    obstacles = {
        "LEFT": ["obstacle"] if values[0] else [],
        "CENTRE": ["obstacle"] if values[1] else [],
        "RIGHT": ["obstacle"] if values[2] else []
    }

    result = decide_movement_direction(obstacles)

    print(f"{name}: {result}")
print("\n--- MOVEMENT STABILITY TEST ---")

previous_decision = None
consecutive_count = 0

simulated_decisions = [
    "MOVE RIGHT",
    "MOVE RIGHT",
    "NO OBSTACLE DETECTED",
    "MOVE LEFT",
    "MOVE LEFT",
    "STOP",
    "MOVE RIGHT",
    "MOVE RIGHT"
]

for decision in simulated_decisions:

    stable_decision, previous_decision, consecutive_count = (
        stabilize_movement_decision(
            decision,
            previous_decision,
            consecutive_count
        )
    )

    print(
        f"Input: {decision:20} | Output: {stable_decision}"
    )