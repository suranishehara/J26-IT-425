"""Fixed attribute vocabulary for the first zero-shot research baseline."""
TAXONOMY_VERSION = "1.0"
# Descriptions intentionally separate garment construction from fabric techniques.
ATTRIBUTES = {
    "garment_type": {
        "shirt": "a button-front shirt",
        "blouse": "a women's blouse",
        "kurta": "a long South Asian kurta tunic",
        "dress": "a one-piece dress with a skirt",
        "saree": "a South Asian saree with a blouse and draped fabric",
        "trousers": "a pair of trousers",
        "skirt": "a separate skirt",
        "other": "an accessory or an object that is not a garment",
    },
    "pattern_type": {
        "solid": "plain solid-coloured fabric without a print",
        "striped": "fabric with stripes",
        "checked": "fabric with a check or plaid pattern",
        "floral": "fabric with a floral print",
        "geometric": "fabric with a geometric print",
        "abstract": "fabric with abstract patterns",
        "other": "fabric with an unrecognizable or other pattern",
    },
    "drape_style": {
        "not_applicable": "a tailored shirt, blouse, kurta or dress, not a draped saree",
        "nivi": "a saree with front skirt pleats and a long pallu over one shoulder",
        "kandyan_osariya": "a Sri Lankan Kandyan Osariya saree with a frill around the waist",
        "other": "another draped garment or a saree with an unclear draping style",
    },
    "fabric_texture": {
        "smooth_matte": "fabric with a smooth matte surface",
        "shiny": "fabric with a shiny lustrous surface",
        "coarse_woven": "fabric with a coarse visibly woven surface",
        "knitted": "fabric with a knitted surface",
        "lace": "fabric with an open lace surface",
        "other": "fabric whose surface texture is unclear",
    },
}
PROMPT_TEMPLATES = ("A product photo of {description}.",
                    "A photo showing clothing: {description}.")
