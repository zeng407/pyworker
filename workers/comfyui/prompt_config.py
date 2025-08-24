# prompt_config.py

styles = {
    "style_eu1": {
        "img": "workers/comfyui/misc/style_eu1.jpg",
        "prompts": [
            ("Scandinavian-inspired minimalist interior with clean lines, light wood accents, geometric forms, and practical furnishings", 1.1),
            ("Soft neutral palette with pops of bright, cheerful color to create warmth and contrast in a cold climate", 1.0),
            ("Emphasis on simplicity, functionality, and airy open spaces, with natural materials and understated decor", 1.0),
        ],
    },
    "style_jp1": {
        "img": "style_jp1.png",
        "prompts": [
            ("Traditional Japanese interior with warm dark wood tones, natural materials, clean symmetry and quiet order, inspired by wabi-sabi aesthetics", 1.1),
            ("Shoji paper doors, wooden lattice screens, tatami mats, minimal furnishings", 1.0),
            ("Harmonious connection with natural light and surrounding landscape, understated elegance, rustic textures", 1.0),
        ],
    },
    "style_country": {
        "img": "style_country.jpg",
        "prompts": [
            ("Vintage English country house style, with aged wooden furniture, floral and botanical patterns, natural materials, warm and nostalgic atmosphere", 1.1),
            ("Exposed wooden beams, plaster walls, terracotta or stone flooring", 1.0),
            ("Cast iron or candle-style light fixtures, cozy and rustic textures, old-world charm", 1.0),
        ],
    },
}

room_prompt = {
    "living_room": {
        "positive": [
            ("A cozy and functional living room with a realistic layout", 1.3),
            ("Featuring sofa set or sectional, coffee table, TV with TV stand or console, media storage, side tables, area rug, floor or table lamps, optional accent chairs", 1.3),
            ("Furniture arranged for comfort and natural traffic flow, walls used efficiently for storage or entertainment units", 1.2),
            ("Clutter-free space with moderate decorations like plants or framed artwork", 1.1),
        ],
        "negative": [
            ("dining table", 1.1), ("dining chairs", 1.1), ("bed", 1.1), ("wardrobe", 1.1),
            ("toilet", 1.1), ("bathtub", 1.1), ("stove", 1.1), ("refrigerator", 1.1)
        ]
    },
    "dining_room": {
        "positive": [
            ("A well-lit dining room with a practical arrangement", 1.3),
            ("Featuring central dining table with chairs, overhead pendant lighting, sideboard or buffet cabinet, optional bar cart or wine rack", 1.3),
            ("Furniture positioned for easy movement", 1.2),
            ("Neatly arranged tableware or decorative centerpieces", 1.1),
        ],
        "negative": [
            ("sofa", 1.1), ("TV", 1.1), ("bed", 1.1), ("bathtub", 1.1),
            ("stove", 1.1), ("refrigerator", 1.1), ("study desk", 1.1)
        ]
    },
    "bedroom": {
        "positive": [
            ("A tranquil bedroom designed for rest", 1.3),
            ("Featuring bed with headboard, bedside tables with lamps, wardrobe or closet, dresser or chest of drawers, optional desk or vanity", 1.3),
            ("Furniture realistically placed to optimize space", 1.2),
            ("Minimal decor like framed pictures or small plants", 1.1),
        ],
        "negative": [
            ("sofa", 1.1), ("dining table", 1.1), ("TV", 1.1),
            ("stove", 1.1), ("refrigerator", 1.1), ("toilet", 1.1)
        ]
    },
    "bathroom": {
        "positive": [
            ("A clean, serene bathroom with a functional layout", 1.3),
            ("Featuring sink with vanity and mirror, toilet, shower or bathtub with glass partition or curtain, towel racks, storage shelves or cabinets", 1.3),
            ("Proper lighting and dry surfaces", 1.2),
            ("Minimal visible items such as soap dispensers or toothbrush holders", 1.1),
        ],
        "negative": [
            ("sofa", 1.1), ("bed", 1.1), ("dining table", 1.1),
            ("stove", 1.1), ("refrigerator", 1.1)
        ]
    },
    "balcony": {
        "positive": [
            ("A small but charming balcony with a view", 1.3),
            ("Featuring outdoor furniture such as small table with chairs or a bench, potted plants, railing decorations", 1.3),
            ("Simple and inviting layout", 1.2),
            ("Clear floor space for movement", 1.1),
        ],
        "negative": [
            ("sofa", 1.1), ("TV", 1.1), ("dining table", 1.1),
            ("bed", 1.1), ("wardrobe", 1.1), ("bathtub", 1.1)
        ]
    },
    "kitchen": {
        "positive": [
            ("A practical and tidy modern kitchen with an efficient layout", 1.3),
            ("Featuring sink with countertop, range hood, induction stove or gas stove, dishwasher, refrigerator, built-in oven, microwave, upper and lower storage cabinets, under-cabinet lighting", 1.4),
            ("Well-organized furniture arranged realistically along walls or corners", 1.2),
            ("Clean floor and clutter-free surfaces", 1.1),
        ],
        "negative": [
            ("wooden furniture", 1.4),
            ("wooden cabinets", 1.4),
            ("wooden countertop", 1.4),
            ("sofa", 1.1),
            ("bed", 1.1),
            ("dining table", 1.1),
            ("wardrobe", 1.1),
            ("bathtub", 1.1),
            ("study desk", 1.1)
        ]
    },
    "study": {
        "positive": [
            ("A quiet study area with a functional layout", 1.3),
            ("Featuring work desk with comfortable chair, bookshelves, task lighting, filing cabinets or storage units", 1.3),
            ("Optional decorative items like plants or framed prints", 1.2),
            ("Organized and clutter-free workspace", 1.1),
        ],
        "negative": [
            ("sofa", 1.1), ("bed", 1.1), ("dining table", 1.1),
            ("stove", 1.1), ("refrigerator", 1.1)
        ]
    },
    "office": {
        "positive": [
            ("A clean and focused home office setup", 1.3),
            ("Featuring large desk, ergonomic office chair, bookshelves or storage cabinets, computer with peripherals", 1.3),
            ("Task lighting and optional whiteboard or pinboard", 1.2),
            ("Minimal clutter for productivity", 1.1),
        ],
        "negative": [
            ("sofa", 1.1), ("bed", 1.1), ("dining table", 1.1),
            ("stove", 1.1), ("refrigerator", 1.1)
        ]
    }
}

common_negative_prompts = [
    ("cartoon", 1.2), ("anime", 1.2), ("painting", 1.2), ("3D render", 1.2),
    ("CGI", 1.2), ("digital art", 1.2), ("illustration", 1.2),
    ("blurry", 1.3), ("noisy", 1.2), ("low quality", 1.3), ("lowres", 1.3),
    ("deformed", 1.3), ("disfigured", 1.3), ("unrealistic", 1.2),
    ("surreal", 1.1), ("fantasy elements", 1.1), ("distorted", 1.2),
    ("exaggerated proportions", 1.2), ("sketch", 1.1),
    ("airbrushed", 1.1), ("overexposed", 1.1), ("underexposed", 1.1),
    ("overly artistic", 1.0), ("HDR", 1.0), ("sci-fi", 1.0),
    ("futuristic", 1.0), ("game style", 1.0), ("virtual reality", 1.0),
    ("noisy textures", 1.1), ("artifacts", 1.1),
    ("window", 1.0), ("floor-to-ceiling window", 1.0), ("glass panel", 1.0),
    ("outdoor view", 1.0), ("reflections", 1.0), ("transparent surfaces", 1.0),
    ("balcony", 1.0), ("curtain wall", 1.0), ("large window", 1.0),
    ("glass facade", 1.0), ("full-height glazing", 1.0), ("glazed opening", 1.0)
]
