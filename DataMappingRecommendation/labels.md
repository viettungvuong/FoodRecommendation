# Recipe Classification Labels

The rows are USDA FDC food items (10,845), not full recipes: each has a description, `product` / `adj` / `verb` terms and nine nutrient columns. `recipe_labeling.ipynb` turns these into multi-label silver labels with keyword + nutrient rules, and `recipe_classification.ipynb` trains a classifier on the terms and nutrients to predict them.

A keyword list is only a starting point for these labels. Each depends on context: the combination of terms, the preparation (verbs), or the nutrient balance. The classifier learns that context from the term embeddings and nutrient components, so it can label items the rules miss.

---

## Label Overview

Rows = rows with the label in `input_stage2/recipe_classes.csv`. An item can have several labels.

| Group | Label | Column | Rows |
|---|---|---|---:|
| Cuisine & Style | Fusion Cuisine | `fusion_cuisine` | 55 |
| | Street Food Style | `street_food_style` | 556 |
| Occasion & Meal Role | Weeknight Dinner | `weeknight_dinner` | 626 |
| | Kid-Friendly | `kid_friendly` | 2,102 |
| | Breakfast | `breakfast` | 675 |
| | Dessert / Sweet Treat | `dessert_sweet_treat` | 391 |
| | Protein Centerpiece | `protein_centerpiece` | 2,247 |
| Flavor | Spicy | `spicy` | 72 |
| | Umami-Rich | `umami_rich` | 1,785 |
| | Smoky / Grilled | `smoky_grilled` | 953 |
| Effort & Skill | Advanced / Gourmet | `advanced_gourmet` | 168 |
| | Easy | `easy` | 4,319 |
| | One-Pot / One-Pan | `one_pot_one_pan` | 505 |
| | Ready-to-Eat / Convenience | `ready_to_eat` | 1,846 |
| Texture & Satiety | Crispy / Crunchy | `crispy_crunchy` | 859 |
| | Hearty / Filling | `hearty_filling` | 2,394 |
| — | Generic (no other label) | `generic` | 2,648 |

`fusion_cuisine` is still labelled but is too rare to learn, so `recipe_classification.ipynb` drops it.

---

## 1. Cuisine & Style
*These labels depend on how ingredients are combined and served, not only on geographic keywords.*

*   **Fusion Cuisine (e.g., Tex-Mex, Asian-Mexican Fusion):** Rules cannot easily tell where one cuisine ends and another begins from raw text alone.
*   **Street Food Style:** Depends on portability, preparation speed and cultural presentation rather than an ingredient checklist.

## 2. Occasion & Meal Role
*When an item is eaten and what part of the meal it plays. Keywords give a starting point, but the nutrient profile decides edge cases (e.g. "chicken" in a soup vs a plain breast).*

*   **Weeknight Dinner:** Depends on prep time, cook time, tool usage and overall effort, which simple text filters cannot measure.
*   **Kid-Friendly:** Relies on the absence of polarizing textures and strong flavors, which rules struggle to evaluate.
*   **Breakfast:** Items eaten in the morning: cereal, oatmeal, pancakes, waffles, eggs, bacon, muffins, bagels, yogurt. Eggs and bacon also appear in other meals, so this needs the combination of product, preparation (`scrambled`, `toasted`) and nutrients.
*   **Dessert / Sweet Treat:** Cakes, cookies, pies, puddings, ice cream and candy. Sugar content separates a real dessert from a savory pie or an unsweetened baking ingredient.
*   **Protein Centerpiece:** A meat, poultry, fish or seafood item a meal is built around. The product term names the cut, and the nutrients confirm it (high protein, almost no carbohydrate). Breaded, sauced and mixed dishes fall out on the nutrient profile.

## 3. Flavor
*Flavors come from how ingredients amplify, mask or balance one another, and from how the food is cooked.*

*   **Spicy:** Food with chili, mustard or related products (like mala, wasabi, horseradish, sriracha, kimchi) in the product terms. Mild variants and sweet peppers are excluded.
*   **Umami-Rich:** Requires detecting the combined effect of specific ingredients (like tomatoes + mushrooms + soy sauce) that create deep savory notes.
*   **Smoky / Grilled:** Food cooked with dry heat (grilled, broiled, roasted, smoked, barbecued, rotisserie). The flavor comes from the cooking method (the verbs), not the ingredient.

## 4. Effort & Skill
*Complexity and convenience live in the verbs (`frozen`, `canned`, `braised`), not in the product names.*

*   **Advanced / Gourmet:** A dish with few ingredients (like a French soufflé) can still require technical precision that keyword rules miss.
*   **Easy:** Easy-to-find ingredients and easy to cook, without much prep time: ready-to-eat items, fresh produce, and simply cooked foods (boiled, baked, scrambled). Slow methods (braised, whole roasts), gourmet items and hard-to-find ingredients (game meat, imported cuts, specialty seafood) are excluded.
*   **One-Pot / One-Pan:** Rules looking for "pot" or "pan" fail when a dish needs separate components (breaded, layered, stuffed, served with sides).
*   **Ready-to-Eat / Convenience:** Frozen meals, canned goods, fast food, restaurant items, snacks and anything labelled `ready`, `instant` or `prepared`. Verb and adjective terms are the main signal, backed by high sodium for processed items.

## 5. Texture & Satiety
*Sensory and "how filling" qualities that come from the macro balance rather than a named ingredient.*

*   **Crispy / Crunchy:** Fried, toasted or baked-dry foods (chips, crackers, pretzels, fried chicken, nuts, granola). A frying or toasting verb plus a fat + carbohydrate-dense profile. A rule on "fried" alone would miss crackers and nuts and wrongly include fried eggs or fried rice.
*   **Hearty / Filling:** High satiety value: lots of protein and fiber per calorie (lean meats, fish, beans, eggs). This is a nutrient ratio, so it is learned from the nutrient components rather than from text.

---

## Label Rules

- **Generic:** rows with no other label are `generic`. It is exclusive: a `generic` row never has another label.
- **Item types (not labels):** the labeling notebook still detects drinks, pantry ingredients (oils, spices, flour, dry grains / beans, dry mixes) and raw fresh produce, only to include or exclude rows. Drinks never get `crispy_crunchy`, `easy`, `ready_to_eat`, `hearty_filling` or `smoky_grilled`. Pantry ingredients never get `easy`, `ready_to_eat` or `hearty_filling`. Fresh produce counts as `easy`. Rows that are only one of these item types end up `generic`.
- **Metadata (not targets):** `heat_level` (`none` / `medium` / `hot`, from the spicy products) and `cuisine_region` (one or more regions, pipe-joined) are stored alongside the labels.
- **Caveat:** these are silver labels from rules over the same inputs the classifier sees, so model scores measure how well the rules are recovered, not true label quality.
