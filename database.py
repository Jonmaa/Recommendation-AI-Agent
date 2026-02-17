# ============================================================
# database.py - Product Catalog & User Purchase History
# ============================================================
# Simulates a product database and user purchase records.
# Products span multiple categories: footwear, fitness,
# sports, food, and technology.
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List
import re
import os


@dataclass
class Product:
    """Represents a product in the catalog."""
    id: str
    name: str
    category: str
    description: str
    price: float
    tags: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Return a rich text representation used for embedding generation."""
        return (
            f"{self.name}. Category: {self.category}. "
            f"{self.description} "
            f"Tags: {', '.join(self.tags)}."
        )


@dataclass
class UserPurchase:
    """Represents a user's purchase history."""
    user_id: str
    username: str
    purchased_product_ids: List[str] = field(default_factory=list)


# ----------------------------------------------------------------
# Product Catalog
# ----------------------------------------------------------------

PRODUCTS: Dict[str, Product] = {}

_raw_products = [
    # --- Footwear ---
    Product("P001", "Nike Air Max 90", "Footwear",
            "Classic running sneakers with visible Air cushioning and retro design.",
            129.99, ["sneakers", "running", "nike", "casual"]),
    Product("P002", "Adidas Ultraboost 22", "Footwear",
            "High-performance running shoes with responsive Boost midsole.",
            189.99, ["sneakers", "running", "adidas", "performance"]),
    Product("P003", "New Balance 574", "Footwear",
            "Iconic lifestyle sneakers combining comfort and heritage style.",
            89.99, ["sneakers", "lifestyle", "new balance", "casual"]),
    Product("P004", "Puma RS-X", "Footwear",
            "Bold chunky sneakers with retro-futuristic design.",
            109.99, ["sneakers", "lifestyle", "puma", "chunky"]),
    Product("P005", "Nike Metcon 8", "Footwear",
            "Cross-training shoes built for heavy lifts and HIIT workouts.",
            139.99, ["training", "crossfit", "nike", "gym"]),

    # --- Fitness Equipment ---
    Product("P006", "Hex Dumbbell Set 5–25 kg", "Fitness",
            "Rubber-coated hex dumbbells with ergonomic grip, set of 5 pairs.",
            249.99, ["weights", "dumbbells", "strength", "home gym"]),
    Product("P007", "Adjustable Kettlebell 4–18 kg", "Fitness",
            "Space-saving adjustable kettlebell for full-body workouts.",
            89.99, ["kettlebell", "weights", "functional", "home gym"]),
    Product("P008", "Olympic Barbell 20 kg", "Fitness",
            "Competition-grade 20 kg Olympic barbell with knurled grip.",
            199.99, ["barbell", "weights", "powerlifting", "strength"]),
    Product("P009", "Resistance Bands Set (5 levels)", "Fitness",
            "Latex resistance bands from light to extra-heavy for versatile training.",
            29.99, ["bands", "resistance", "rehab", "portable"]),
    Product("P010", "Yoga Mat Premium 6mm", "Fitness",
            "Non-slip TPE yoga mat with alignment guides, eco-friendly.",
            49.99, ["yoga", "mat", "flexibility", "pilates"]),

    # --- Sports ---
    Product("P011", "Adidas UEFA Champions League Ball", "Sports",
            "Official match ball with seamless surface for top-level play.",
            49.99, ["football", "soccer", "ball", "adidas"]),
    Product("P012", "Spalding NBA Official Basketball", "Sports",
            "Full-grain leather basketball used in NBA competition.",
            159.99, ["basketball", "ball", "spalding", "nba"]),
    Product("P013", "Wilson US Open Tennis Ball (4-pack)", "Sports",
            "Tournament-grade tennis balls with extra-duty felt.",
            9.99, ["tennis", "ball", "wilson", "tournament"]),
    Product("P014", "Mikasa V200W Volleyball", "Sports",
            "FIVB-approved match volleyball with micro-fiber cover.",
            69.99, ["volleyball", "ball", "mikasa", "competition"]),
    Product("P015", "Everlast Pro Boxing Gloves 12 oz", "Sports",
            "Premium synthetic leather boxing gloves with wrist support.",
            59.99, ["boxing", "gloves", "combat", "training"]),

    # --- Food & Nutrition ---
    Product("P016", "Whey Protein Isolate 2 kg (Chocolate)", "Food",
            "Low-fat whey protein isolate with 30 g protein per serving.",
            54.99, ["protein", "supplement", "chocolate", "fitness"]),
    Product("P017", "Organic Granola Mix 1 kg", "Food",
            "Crunchy granola with oats, nuts, and dried fruits, no added sugar.",
            12.99, ["granola", "breakfast", "organic", "healthy"]),
    Product("P018", "Energy Bar Variety Pack (12 bars)", "Food",
            "Mixed flavors energy bars with natural ingredients for quick fuel.",
            24.99, ["energy", "bars", "snack", "portable"]),
    Product("P019", "Cold-Pressed Olive Oil 1L", "Food",
            "Extra virgin olive oil from Mediterranean olives, first cold press.",
            14.99, ["olive oil", "cooking", "mediterranean", "healthy"]),
    Product("P020", "Organic Quinoa 500g", "Food",
            "Tri-color organic quinoa, high in protein and gluten-free.",
            8.99, ["quinoa", "grain", "organic", "superfood"]),
    Product("P021", "Matcha Green Tea Powder 100g", "Food",
            "Ceremonial-grade Japanese matcha, rich in antioxidants.",
            19.99, ["matcha", "tea", "japanese", "antioxidant"]),
    Product("P022", "Creatine Monohydrate 500g", "Food",
            "Micronized creatine monohydrate for strength and power output.",
            19.99, ["creatine", "supplement", "performance", "strength"]),

    # --- Technology ---
    Product("P023", "Apple AirPods Pro 2", "Technology",
            "Wireless earbuds with active noise cancellation and spatial audio.",
            249.99, ["earbuds", "wireless", "apple", "audio"]),
    Product("P024", "Garmin Forerunner 265", "Technology",
            "GPS running watch with AMOLED display and training metrics.",
            449.99, ["smartwatch", "gps", "garmin", "running"]),
    Product("P025", "Fitbit Charge 6", "Technology",
            "Fitness tracker with heart rate, sleep tracking, and Google integration.",
            159.99, ["fitness tracker", "wearable", "fitbit", "health"]),
    Product("P026", "Sony WH-1000XM5 Headphones", "Technology",
            "Over-ear wireless headphones with industry-leading noise cancellation.",
            349.99, ["headphones", "wireless", "sony", "noise cancelling"]),
    Product("P027", "GoPro HERO 12 Black", "Technology",
            "Waterproof action camera with 5.3K video and HyperSmooth stabilization.",
            399.99, ["camera", "action", "gopro", "video"]),
    Product("P028", "Kindle Paperwhite 2024", "Technology",
            "E-reader with 6.8\" glare-free display and adjustable warm light.",
            139.99, ["ereader", "kindle", "amazon", "books"]),
    Product("P029", "Logitech MX Master 3S Mouse", "Technology",
            "Ergonomic wireless mouse with MagSpeed scroll and multi-device support.",
            99.99, ["mouse", "ergonomic", "logitech", "productivity"]),
    Product("P030", "Samsung Galaxy Tab S9", "Technology",
            "11-inch Android tablet with AMOLED display and S Pen included.",
            749.99, ["tablet", "samsung", "android", "productivity"]),
]

for p in _raw_products:
    PRODUCTS[p.id] = p


# ----------------------------------------------------------------
# Simulated User Purchase Histories
# ----------------------------------------------------------------

USER_PURCHASES: List[UserPurchase] = [
    UserPurchase("U001", "Alex", ["P001", "P005", "P006", "P016", "P022", "P024"]),
    UserPurchase("U002", "Maria", ["P001", "P002", "P018", "P024", "P025"]),
    UserPurchase("U003", "James", ["P001", "P003", "P017", "P019", "P028"]),
    UserPurchase("U004", "Sofia", ["P005", "P006", "P007", "P008", "P016", "P022"]),
    UserPurchase("U005", "Carlos", ["P001", "P011", "P012", "P016", "P018"]),
    UserPurchase("U006", "Emma", ["P002", "P023", "P024", "P026", "P027"]),
    UserPurchase("U007", "Liam", ["P003", "P009", "P010", "P017", "P021"]),
    UserPurchase("U008", "Olivia", ["P001", "P004", "P012", "P015", "P016"]),
    UserPurchase("U009", "Noah", ["P005", "P006", "P007", "P008", "P009", "P016", "P022"]),
    UserPurchase("U010", "Ava", ["P023", "P026", "P028", "P029", "P030", "P021"]),
    UserPurchase("U011", "Ethan", ["P001", "P002", "P011", "P013", "P014", "P025"]),
    UserPurchase("U012", "Mia", ["P004", "P015", "P009", "P016", "P018", "P022"]),
    UserPurchase("U013", "Lucas", ["P002", "P024", "P027", "P018", "P005"]),
    UserPurchase("U014", "Isabella", ["P010", "P017", "P019", "P020", "P021", "P028"]),
    UserPurchase("U015", "Mason", ["P001", "P002", "P003", "P004", "P023", "P026"]),
    UserPurchase("U016", "Jon", ["P028", "P027", "P017"]),
    UserPurchase("U017", "Albert", ["P001"]),
]


def get_product(product_id: str) -> Product | None:
    """Retrieve a product by its ID."""
    return PRODUCTS.get(product_id)


def get_all_products() -> List[Product]:
    """Return all products in the catalog."""
    return list(PRODUCTS.values())


def get_co_purchased_products(product_id: str) -> Dict[str, int]:
    """
    Find products frequently bought together with the given product.
    Returns a dict mapping product_id → count of co-purchases.
    """
    co_purchase_counts: Dict[str, int] = {}

    for user in USER_PURCHASES:
        if product_id in user.purchased_product_ids:
            for other_id in user.purchased_product_ids:
                if other_id != product_id:
                    co_purchase_counts[other_id] = co_purchase_counts.get(other_id, 0) + 1

    # Sort descending by co-purchase count
    return dict(sorted(co_purchase_counts.items(), key=lambda x: x[1], reverse=True))


def get_users_who_bought(product_id: str) -> List[UserPurchase]:
    """Return all users who purchased a specific product."""
    return [u for u in USER_PURCHASES if product_id in u.purchased_product_ids]


def get_user_by_name(username: str) -> UserPurchase | None:
    """Find an existing user by their username (case-insensitive)."""
    for user in USER_PURCHASES:
        if user.username.lower() == username.lower():
            return user
    return None


def _generate_user_id() -> str:
    """Generate the next available user ID (e.g., U016, U017, ...)."""
    max_num = 0
    for user in USER_PURCHASES:
        num = int(user.user_id[1:])  # Extract number from "U001"
        if num > max_num:
            max_num = num
    return f"U{max_num + 1:03d}"


def add_purchase(username: str, product_id: str) -> UserPurchase:
    """
    Record a new purchase. Looks up the user by name.
    If the user doesn't exist, creates a new one with an auto-generated ID.
    Returns the user with the updated purchase list.
    Automatically persists changes to this source file.
    """
    user = get_user_by_name(username)

    if user:
        if product_id not in user.purchased_product_ids:
            user.purchased_product_ids.append(product_id)
        _save_purchases_to_source()
        return user

    new_id = _generate_user_id()
    new_user = UserPurchase(new_id, username, [product_id])
    USER_PURCHASES.append(new_user)
    _save_purchases_to_source()
    return new_user


# ----------------------------------------------------------------
# Persistence — rewrite USER_PURCHASES block in this source file
# ----------------------------------------------------------------

_THIS_FILE = os.path.abspath(__file__)


def _save_purchases_to_source() -> None:
    """
    Rewrite the USER_PURCHASES list in database.py so that new
    users and purchases persist across restarts without any
    external files.
    """
    # Build the new USER_PURCHASES block
    lines = [
        "USER_PURCHASES: List[UserPurchase] = [",
    ]
    for user in USER_PURCHASES:
        ids_str = ", ".join(f'"{pid}"' for pid in user.purchased_product_ids)
        lines.append(
            f'    UserPurchase("{user.user_id}", "{user.username}", [{ids_str}]),'
        )
    lines.append("]")
    new_block = "\n".join(lines)

    # Read current file content
    with open(_THIS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the USER_PURCHASES block using regex
    # Matches from "USER_PURCHASES: List[UserPurchase] = [" to the closing "]"
    pattern = r"USER_PURCHASES: List\[UserPurchase\] = \[.*?^\]"
    new_content = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL | re.MULTILINE)

    # Write back
    with open(_THIS_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)
