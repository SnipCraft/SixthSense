
# obstacle_prioritization.py
# SixthSense
# Obstacle Prioritization & Distance-Level Classification


# --------------------------------------------------
# 1. THRESHOLDS
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.50

# Experimental relative-depth thresholds
CLOSE_THRESHOLD = 2.85
MEDIUM_THRESHOLD = 2.51


# --------------------------------------------------
# 2. DISTANCE-LEVEL CLASSIFICATION
# --------------------------------------------------

def detect_distance_level(depth_score):
    """
    Classify an obstacle using its relative depth score.

    CLOSE  : score >= 2.85
    MEDIUM : score >= 2.51 and score < 2.85
    FAR    : score < 2.51

    These are experimental relative-depth categories,
    not physical distances in metres.
    """

    depth_score = float(depth_score)

    if depth_score >= CLOSE_THRESHOLD:
        return "CLOSE"

    elif depth_score >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    else:
        return "FAR"


# --------------------------------------------------
# 3. CALCULATE PRIORITY SCORE
# --------------------------------------------------

def calculate_priority(obstacle):
    """
    Assign priority according to distance level.

    CLOSE  = 3
    MEDIUM = 2
    FAR    = 1
    """

    distance_level = obstacle["distance_level"]

    priority_map = {
        "CLOSE": 3,
        "MEDIUM": 2,
        "FAR": 1
    }

    return priority_map.get(distance_level, 0)


# --------------------------------------------------
# 4. GENERATE ALERT
# --------------------------------------------------

def generate_alert(obstacle):
    """
    Generate an alert according to distance level.
    """

    name = obstacle["name"]
    direction = obstacle["direction"].upper()

    depth_score = float(obstacle["relative_depth"])

    distance_level = obstacle["distance_level"]

    if distance_level == "CLOSE":
        message = "WARNING: Close obstacle detected"

    elif distance_level == "MEDIUM":
        message = "Obstacle at medium relative depth"

    else:
        message = "FAR obstacle detected"

    return (
        f"{message}: {name} "
        f"on the {direction} "
        f"(Relative depth score: {depth_score:.3f})"
    )


# --------------------------------------------------
# 5. PRIORITIZE ALL OBSTACLES
# --------------------------------------------------

def prioritize_obstacles(obstacles):
    """
    Filter, classify and prioritize ALL valid detections.

    No top-3 or top-5 limit is applied here.

    Output fields:
        distance_level
        priority_score
        priority_rank
        priority
        is_primary
        alert
    """

    valid_obstacles = []

    # ----------------------------------------------
    # Filter and classify detections
    # ----------------------------------------------

    for obstacle in obstacles:

        confidence = float(
            obstacle.get("confidence", 0.0)
        )

        # Ignore low-confidence detections
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Copy original detection data
        processed_obstacle = obstacle.copy()

        # Classify relative depth
        processed_obstacle["distance_level"] = (
            detect_distance_level(
                processed_obstacle["relative_depth"]
            )
        )

        # Calculate priority
        processed_obstacle["priority_score"] = (
            calculate_priority(
                processed_obstacle
            )
        )

        valid_obstacles.append(processed_obstacle)

    # ----------------------------------------------
    # Sort obstacles by priority
    # ----------------------------------------------

    # CLOSE first, then MEDIUM, then FAR.
    # Within the same level, higher relative-depth
    # score comes first.

    valid_obstacles.sort(
        key=lambda item: (
            item["priority_score"],
            float(item["relative_depth"])
        ),
        reverse=True
    )

    # ----------------------------------------------
    # Assign ranks and generate alerts
    # ----------------------------------------------

    for rank, obstacle in enumerate(
        valid_obstacles,
        start=1
    ):

        obstacle["priority_rank"] = rank

        # Compatibility with existing test script
        obstacle["priority"] = rank

        # Highest-ranked obstacle
        obstacle["is_primary"] = (rank == 1)

        obstacle["alert"] = generate_alert(
            obstacle
        )

    # Return every valid prioritized obstacle
    return valid_obstacles


# --------------------------------------------------
# 6. SAMPLE TEST
# --------------------------------------------------

if __name__ == "__main__":

    detected_obstacles = [

        {
            "name": "person",
            "direction": "CENTRE",
            "relative_depth": 3.559,
            "confidence": 0.84
        },

        {
            "name": "bottle",
            "direction": "RIGHT",
            "relative_depth": 4.450,
            "confidence": 0.88
        },

        {
            "name": "chair",
            "direction": "CENTRE",
            "relative_depth": 2.700,
            "confidence": 0.75
        },

        {
            "name": "backpack",
            "direction": "LEFT",
            "relative_depth": 3.100,
            "confidence": 0.81
        },

        {
            "name": "cell phone",
            "direction": "RIGHT",
            "relative_depth": 3.500,
            "confidence": 0.90
        }

    ]

    results = prioritize_obstacles(
        detected_obstacles
    )

    print(
        "\n===== SIXTHSENSE OBSTACLE PRIORITIZATION =====\n"
    )

    print("Total valid obstacles:", len(results))
    print("-----------------------------------")

    if not results:
        print("No valid obstacles detected.")

    for obstacle in results:

        print("Object:", obstacle["name"])

        print("Direction:", obstacle["direction"])

        print("Confidence:", obstacle["confidence"])

        print("Relative Depth:", obstacle["relative_depth"])

        print("Distance Level:", obstacle["distance_level"])

        print("Priority Score:", obstacle["priority_score"])

        print("Priority Rank:", obstacle["priority_rank"])

        print("Primary Obstacle:", obstacle["is_primary"])

        print("Alert:", obstacle["alert"])

        print("-----------------------------------")
