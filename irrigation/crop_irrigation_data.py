# crop_irrigation_data.py

CROP_IRRIGATION_MAP = {
    # High-value horticulture crops (always drip)
    "tomato": {
        "preferred": ["Drip Irrigation"],
        "notes": "Tomato requires controlled, root-zone irrigation."
    },
    "chili": {
        "preferred": ["Drip Irrigation"],
        "notes": "Maintains uniform moisture critical for chili."
    },
    "grapes": {
        "preferred": ["Drip Irrigation"],
        "notes": "Essential for vineyard moisture management."
    },
    "banana": {
        "preferred": ["Drip Irrigation", "Basin Irrigation"],
        "notes": "Banana responds well to drip; basin used traditionally."
    },

    # Field crops
    "cotton": {
        "preferred": ["Drip Irrigation", "Furrow Irrigation"],
        "notes": "Drip saves water; furrow suitable for black soil cotton belts."
    },
    "sugarcane": {
        "preferred": ["Drip Irrigation", "Furrow Irrigation"],
        "notes": "Drip increases yield; furrow widely adopted."
    },
    "maize": {
        "preferred": ["Sprinkler", "Furrow Irrigation"],
        "notes": "Prefers uniform moisture early; furrow later."
    },

    # Cereals
    "wheat": {
        "preferred": ["Sprinkler", "Flood"],
        "notes": "Sprinkler preferred; flood common in Indo-Gangetic belts."
    },
    "rice": {
        "preferred": ["Flood Irrigation", "Basin Irrigation"],
        "notes": "Rice requires standing water (puddled fields)."
    },

    # Pulses
    "pigeon pea": {
        "preferred": ["Furrow Irrigation"],
        "notes": "Requires moderate moisture; avoid waterlogging."
    },
    "green gram": {
        "preferred": ["Sprinkler"],
        "notes": "Sensitive to waterlogging; sprinkler ideal."
    },

    # Vegetables
    "potato": {
        "preferred": ["Sprinkler"],
        "notes": "Avoid wetting tubers by flood; sprinkler increases uniformity."
    },
    "onion": {
        "preferred": ["Drip Irrigation"],
        "notes": "Onions require uniform bulb development."
    }
}
