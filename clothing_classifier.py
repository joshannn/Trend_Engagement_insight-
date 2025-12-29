import clip
import torch
from PIL import Image
import os
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# DEVICE (SAFE GPU FALLBACK)
def get_device():
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return "cuda"
    except Exception as e:
        print("CUDA failed, switching to CPU:", e)
    return "cpu"

# EXPANDED CLOTHING TAXONOMY
CLOTHING_TYPES = [
    # Casual - More Specific
    "plain white t-shirt and blue jeans",
    "graphic t-shirt with jeans casual",
    "polo shirt and chinos smart casual",
    "casual button-down shirt untucked with jeans",
    "henley shirt and jeans outfit",
    "flannel plaid shirt casual wear",
    "shorts and tank top summer casual",
    "cargo shorts and t-shirt relaxed fit",
    "bermuda shorts casual outfit",
    
    # Streetwear & Urban
    "hoodie and joggers streetwear style",
    "oversized hoodie street fashion",
    "sweatshirt and track pants urban wear",
    "bomber jacket streetwear outfit",
    "denim jacket with jeans double denim",
    "leather jacket street style edgy",
    
    # Smart Casual
    "blazer with jeans smart casual",
    "sport coat and chinos dressy casual",
    "cardigan sweater layered outfit",
    "v-neck sweater over collared shirt",
    
    # Formal & Business
    "full business suit with tie corporate",
    "two-piece suit professional attire",
    "three-piece suit with vest formal",
    "dress shirt and slacks business casual",
    "blazer and dress pants office wear",
    "tuxedo black tie formal wear",
    
    # Traditional Ethnic
    "kurta pajama traditional south asian",
    "sherwani embroidered wedding attire",
    "dhoti kurta traditional indian",
    "pathani suit ethnic wear",
    "saree draped traditional dress",
    "lehenga choli bridal festive",
    "salwar kameez everyday ethnic",
    "anarkali suit flowing dress",
    
    # Outerwear Specific
    "trench coat classic outerwear",
    "puffer jacket puffy winter coat",
    "down jacket insulated cold weather",
    "peacoat wool double-breasted",
    "parka hooded winter jacket",
    "windbreaker light jacket athletic",
    
    # Athletic & Sportswear
    "gym tank top and shorts workout",
    "athletic leggings and sports bra",
    "running shorts and performance shirt",
    "yoga pants and fitted top activewear",
    "basketball jersey and shorts",
    "football soccer kit uniform",
    "compression wear tight athletic clothing",
    "track suit matching set sporty",
    
    # Dresses - Modest to Formal
    "maxi dress long flowing modest",
    "midi dress knee-length elegant",
    "sundress casual summer floral",
    "shirt dress button-down casual",
    "wrap dress tied waist flattering",
    "a-line dress classic silhouette",
    "fit and flare dress feminine",
    "cocktail dress semi-formal party",
    "evening gown floor-length formal",
    "ball gown princess style formal",
    
    # Dresses - Fitted & Revealing
    "bodycon dress tight fitted curves",
    "mini dress short above knee",
    "micro mini dress very short revealing",
    "bandage dress tight fitted elastic",
    "slip dress satin thin straps",
    "backless dress open back exposed",
    "halter neck dress tied neck bare shoulders",
    "off-shoulder dress exposed shoulders",
    "strapless dress no straps bare shoulders",
    "plunging neckline dress deep v-neck cleavage",
    "low-cut dress revealing chest cleavage",
    "side slit dress leg revealing high slit",
    "sheer dress see-through transparent",
    "mesh dress netted transparent fabric",
    "cut-out dress strategic openings skin showing",
    
    # Tops - Revealing
    "crop top bare midriff stomach showing",
    "tube top strapless bandeau",
    "halter top tied neck back exposed",
    "spaghetti strap camisole thin straps",
    "tank top sleeveless casual",
    "low-cut top plunging neckline cleavage",
    "backless top open back bare",
    "cold shoulder top cutout shoulders",
    "bustier corset-style structured top",
    "bralette lace delicate crop",
    "sheer blouse see-through transparent",
    
    # Skirts & Bottoms
    "pencil skirt fitted knee-length",
    "mini skirt short above knee",
    "micro mini skirt very short revealing",
    "pleated skirt flowy feminine",
    "maxi skirt long flowing",
    "high-waisted skirt vintage style",
    "leather skirt edgy tight",
    "denim skirt casual jean fabric",
    
    # Beachwear & Resort
    "bikini two-piece swimsuit revealing",
    "string bikini minimal coverage skimpy",
    "triangle bikini top tied straps",
    "one-piece swimsuit full coverage",
    "monokini cutout one-piece revealing",
    "swim trunks men shorts beachwear",
    "board shorts surfing beach",
    "cover-up sheer beach dress",
    "sarong wrap beach skirt",
    "rash guard athletic swim shirt",
    
    # Lingerie & Intimates
    "lingerie set bra and panties intimate",
    "lace lingerie delicate underwear",
    "silk lingerie satin sleepwear sensual",
    "teddy bodysuit lingerie one-piece",
    "corset structured lingerie waist cinching",
    "bustier strapless lingerie top",
    "camisole slip dress nightwear",
    "negligee sheer nightgown revealing",
    "bra and underwear basic underwear set",
    "push-up bra padded underwire",
    
    # Sleepwear & Lounge
    "pajama set matching top bottom",
    "nightgown long sleep dress",
    "robe bathrobe loungewear",
    "loungewear comfortable home clothes",
    "sweatpants and hoodie cozy casual",
    
    # Professional Uniforms
    "medical scrubs healthcare uniform",
    "chef whites kitchen uniform",
    "security guard uniform official",
    "police uniform law enforcement",
    "military uniform armed forces",
    "school uniform student dress code",
]

# NON-CLOTHING FILTERS
NON_CLOTHING_FILTERS = [
    "text document with writing",
    "menu card with text",
    "screenshot of text",
    "poster with text and graphics",
    "food dish meal on plate",
    "restaurant food photography",
    "dessert cake pastry",
    "fruits and vegetables",
    "landscape nature scenery",
    "building architecture exterior",
    "car vehicle automobile",
    "animal pet wildlife",
    "abstract art pattern",
]

def is_valid_clothing_image(image_path, model, preprocess, device, filter_features, threshold=0.4):
    """Filter out non-clothing images"""
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ filter_features.T).softmax(dim=-1)
            max_score = similarity.max().item()
            
            # If any non-clothing category scores high, reject
            if max_score > threshold:
                return False, f"non_clothing (score: {max_score:.2f})"
        return True, None
    except Exception as e:
        print(f"Error filtering {image_path}: {e}")
        return False, "error"

def get_clothing_attributes(image_path, model, preprocess, device, text_features):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            scores = similarity.cpu().numpy()[0]
        
        return CLOTHING_TYPES, scores
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None, None

def categorize_clothing(clothing_types, type_scores):
    if clothing_types is None or type_scores is None:
        return "uncategorized"
    
    # Get top 3 matches for better context
    top_indices = np.argsort(type_scores)[-3:][::-1]
    top_type = clothing_types[top_indices[0]].lower()
    second_type = clothing_types[top_indices[1]].lower()
    
    # INTIMATE & REVEALING WEAR (Highest Priority)
    if any(word in top_type for word in ["lingerie", "bra", "panties", "underwear", "teddy", "negligee"]):
        return "lingerie_intimates"
    
    if "corset" in top_type or "bustier" in top_type:
        return "corset_bustier"
    
    if "bikini" in top_type or "string bikini" in top_type:
        return "bikini_swimwear"
    
    if "monokini" in top_type or ("one-piece" in top_type and "swimsuit" in top_type):
        return "swimsuit_onepiece"
    
    # REVEALING DRESSES & TOPS
    if any(word in top_type for word in ["plunging", "low-cut", "cleavage", "deep v-neck"]):
        return "revealing_neckline"
    
    if "backless" in top_type or "open back" in top_type:
        return "backless_clothing"
    
    if "sheer" in top_type or "see-through" in top_type or "mesh" in top_type or "transparent" in top_type:
        return "sheer_transparent"
    
    if "bodycon" in top_type or "bandage dress" in top_type or "tight fitted" in top_type:
        return "bodycon_fitted"
    
    if "micro mini" in top_type or "very short" in top_type:
        return "micro_mini"
    
    if "mini" in top_type and "dress" in top_type:
        return "mini_dress"
    
    if "mini" in top_type and "skirt" in top_type:
        return "mini_skirt"
    
    if "crop top" in top_type or "bare midriff" in top_type:
        return "crop_top"
    
    if "tube top" in top_type or "bandeau" in top_type or "strapless" in top_type:
        return "tube_strapless"
    
    if "bralette" in top_type:
        return "bralette_top"
    
    if "cut-out" in top_type or "cutout" in top_type:
        return "cutout_dress"
    
    if "slit" in top_type and "dress" in top_type:
        return "slit_dress"
    
    # FORMAL WEAR
    if "tuxedo" in top_type or "black tie" in top_type:
        return "tuxedo_formal"
    
    if "ball gown" in top_type or "princess" in top_type:
        return "ball_gown"
    
    if "evening gown" in top_type or "floor-length formal" in top_type:
        return "evening_gown"
    
    if "cocktail dress" in top_type:
        return "cocktail_dress"
    
    if "business suit" in top_type or "corporate" in top_type or ("suit" in top_type and "tie" in top_type):
        return "business_suit"
    
    if "three-piece" in top_type or "vest formal" in top_type:
        return "three_piece_suit"
    
    # TRADITIONAL ETHNIC
    if "sherwani" in top_type or "wedding attire" in top_type:
        return "sherwani_ethnic"
    
    if "kurta pajama" in top_type or "pathani" in top_type:
        return "kurta_traditional"
    
    if "saree" in top_type:
        return "saree_traditional"
    
    if "lehenga" in top_type or "bridal" in top_type:
        return "lehenga_festive"
    
    if "salwar kameez" in top_type or "anarkali" in top_type:
        return "salwar_ethnic"
    
    # DRESSES - MODEST
    if "maxi dress" in top_type or "long flowing" in top_type:
        return "maxi_dress"
    
    if "midi dress" in top_type or "knee-length elegant" in top_type:
        return "midi_dress"
    
    if "sundress" in top_type or "summer floral" in top_type:
        return "sundress_casual"
    
    if "wrap dress" in top_type:
        return "wrap_dress"
    
    if "shirt dress" in top_type:
        return "shirt_dress"
    
    if any(word in top_type for word in ["a-line", "fit and flare"]):
        return "classic_dress"
    
    # SMART CASUAL & BUSINESS CASUAL
    if "blazer" in top_type and "jeans" in top_type:
        return "blazer_jeans_smart"
    
    if "sport coat" in top_type or "chinos dressy" in top_type:
        return "smart_casual"
    
    if "dress shirt" in top_type and "slacks" in top_type:
        return "business_casual"
    
    if "blazer" in top_type and "dress pants" in top_type:
        return "office_wear"
    
    # CASUAL CATEGORIES - MORE SPECIFIC
    if "graphic t-shirt" in top_type:
        return "graphic_tee_casual"
    
    if "polo shirt" in top_type:
        return "polo_casual"
    
    if "flannel" in top_type or "plaid shirt" in top_type:
        return "flannel_casual"
    
    if "henley" in top_type:
        return "henley_casual"
    
    if "plain white t-shirt" in top_type or ("t-shirt" in top_type and "jeans" in top_type):
        return "tshirt_jeans_basic"
    
    if "button-down" in top_type and "casual" in top_type:
        return "casual_buttondown"
    
    if "tank top" in top_type and "shorts" in top_type:
        return "tank_shorts_summer"
    
    if "cargo shorts" in top_type or "bermuda shorts" in top_type:
        return "casual_shorts"
    
    # STREETWEAR & URBAN
    if "hoodie" in top_type and ("oversized" in top_type or "street" in top_type):
        return "hoodie_streetwear"
    
    if "joggers" in top_type or "track pants" in top_type:
        return "joggers_athletic"
    
    if "bomber jacket" in top_type:
        return "bomber_jacket"
    
    if "denim jacket" in top_type or "double denim" in top_type:
        return "denim_jacket"
    
    if "leather jacket" in top_type and "street" in top_type:
        return "leather_jacket_edgy"
    
    # OUTERWEAR
    if "trench coat" in top_type:
        return "trench_coat"
    
    if "puffer" in top_type or "puffy" in top_type or "down jacket" in top_type:
        return "puffer_jacket"
    
    if "parka" in top_type or "hooded winter" in top_type:
        return "parka_winter"
    
    if "peacoat" in top_type or "wool double-breasted" in top_type:
        return "peacoat"
    
    if "windbreaker" in top_type:
        return "windbreaker"
    
    # ATHLETIC & SPORTSWEAR
    if "yoga pants" in top_type or "leggings" in top_type:
        return "yoga_leggings"
    
    if "sports bra" in top_type or "compression" in top_type:
        return "athletic_compression"
    
    if "gym" in top_type or "workout" in top_type:
        return "gym_wear"
    
    if "running" in top_type or "performance" in top_type:
        return "running_gear"
    
    if "basketball" in top_type or "football" in top_type or "soccer" in top_type:
        return "sports_uniform"
    
    if "track suit" in top_type:
        return "tracksuit"
    
    # BEACHWEAR - NON-REVEALING
    if "swim trunks" in top_type or "board shorts" in top_type:
        return "mens_swimwear"
    
    if "rash guard" in top_type:
        return "rash_guard"
    
    if "cover-up" in top_type or "sarong" in top_type:
        return "beach_coverup"
    
    # SKIRTS
    if "pencil skirt" in top_type:
        return "pencil_skirt"
    
    if "pleated skirt" in top_type:
        return "pleated_skirt"
    
    if "leather skirt" in top_type:
        return "leather_skirt"
    
    if "denim skirt" in top_type:
        return "denim_skirt"
    
    if "maxi skirt" in top_type:
        return "maxi_skirt"
    
    # LOUNGEWEAR & SLEEPWEAR
    if "pajama" in top_type or "nightgown" in top_type:
        return "sleepwear"
    
    if "robe" in top_type or "bathrobe" in top_type:
        return "robe_loungewear"
    
    if "loungewear" in top_type or ("sweatpants" in top_type and "hoodie" in top_type):
        return "home_loungewear"
    
    # UNIFORMS
    if "scrubs" in top_type or "healthcare" in top_type:
        return "medical_uniform"
    
    if "chef" in top_type or "kitchen" in top_type:
        return "chef_uniform"
    
    if any(word in top_type for word in ["security", "police", "military"]):
        return "security_uniform"
    
    if "school uniform" in top_type:
        return "school_uniform"
    
    # LAYERING & SWEATERS
    if "cardigan" in top_type or "v-neck sweater" in top_type:
        return "sweater_layered"
    
    # FALLBACK
    return "casual_uncategorized"

def auto_organize_images(folder_path):
    device = get_device()
    print(f"Using device: {device}\n")
    
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("Model loaded\n")
    
    print("Encoding clothing prompts...")
    text_tokens = clip.tokenize(CLOTHING_TYPES).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    print("Encoding filter prompts...")
    filter_tokens = clip.tokenize(NON_CLOTHING_FILTERS).to(device)
    with torch.no_grad():
        filter_features = model.encode_text(filter_tokens)
        filter_features /= filter_features.norm(dim=-1, keepdim=True)
    
    print("Encoding complete\n")
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print("No images found")
        return
    
    results = {}
    counts = defaultdict(int)
    filtered_out = {}
    
    print("Analyzing images...\n")
    for i, img in enumerate(image_files, 1):
        path = os.path.join(folder_path, img)
        
        # First filter non-clothing
        is_valid, reason = is_valid_clothing_image(path, model, preprocess, device, filter_features)
        
        if not is_valid:
            filtered_out[img] = reason
            print(f"[{i}/{len(image_files)}] {img} → FILTERED: {reason}")
            continue
        
        # Then categorize clothing
        types, scores = get_clothing_attributes(path, model, preprocess, device, text_features)
        category = categorize_clothing(types, scores)
        
        results[img] = category
        counts[category] += 1
        
        if types is not None:
            print(f"[{i}/{len(image_files)}] {img} → {category}")
        else:
            print(f"[{i}/{len(image_files)}] {img} → ERROR")
    
    print("\n=== FILTERED OUT ===")
    filter_counts = defaultdict(int)
    for reason in filtered_out.values():
        filter_counts[reason] += 1
    for reason, count in filter_counts.items():
        print(f"{reason}: {count}")
    
    print("\n=== CLOTHING CATEGORIES ===")
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")
    
    choice = input("\nOrganize into folders? (y/n): ").strip().lower()
    if choice not in ("y", "yes"):
        return
    
    # Move filtered images
    if filtered_out:
        filtered_dir = os.path.join(folder_path, "_filtered_out")
        os.makedirs(filtered_dir, exist_ok=True)
        for img in filtered_out.keys():
            shutil.move(
                os.path.join(folder_path, img),
                os.path.join(filtered_dir, img)
            )
    
    # Move categorized images
    for img, cat in results.items():
        dst_dir = os.path.join(folder_path, cat)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(
            os.path.join(folder_path, img),
            os.path.join(dst_dir, img)
        )
    
    print("\n✓ Images organized successfully")

def main():
    folder = input("Enter folder path: ").strip().strip('"').strip("'")
    if not os.path.isdir(folder):
        print("Invalid folder")
        return
    auto_organize_images(folder)

if __name__ == "__main__":
    main()