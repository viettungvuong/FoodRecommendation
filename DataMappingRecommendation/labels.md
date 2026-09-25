# Complex Recipe Classification Labels (Non-Rule-Based)

This document covers recipe classification labels that **cannot be accurately solved using simple keyword filters or if-then rules**. These labels require machine learning models (like classifier chains) because they depend on contextual reasoning, ingredient ratios, technical nuance, and cultural context.

---

## 1. Authenticity, Origin & Fusion
*These labels require analyzing the specific combination and harmony of ingredients rather than just looking for geographic keywords.*

*   **Fusion Cuisine (e.g., Tex-Mex, Asian-Mexican Fusion):** Rules cannot easily distinguish where one cuisine ends and another begins based on raw text alone. 
*   **Street Food Style:** This relies on the context of portability, preparation speed, and cultural presentation rather than an ingredient checklist.

## 2. Occasion, Meal Course & Context
*These labels depend heavily on preparation complexity, presentation, and cultural habits.*

*   **Weeknight Dinner:** Requires evaluating prep time, cook time, tool usage, and overall cognitive load, which simple text filters cannot accurately measure.
*   **Kid-Friendly:** Relies on the subtle absence of polarizing textures, visual appeal, and mild flavor profiles that rules struggle to evaluate.

## 3. Flavor Profiles & Sensory Qualities
*Flavors are chemical interactions. Rules fail because they cannot calculate how ingredients amplify, mask, or balance one another.*

*   **Spicy:** Food with Chili/Mustard or adjenct products (like Mala) in product
*   **Umami-Rich:** Requires detecting the synergistic effect of combining specific ingredients (like tomatoes + mushrooms + soy sauce) that trigger deep savory notes.
*   **Tangy / Zesty:** Relies on the acidic balance against fats and sugars. A rule tracking "lemon juice" will miss the balancing act of sugar that neutralizes sharpness.

## 4. Technical Complexity & Skill Level
*Complexity lives in the verbs of the instructions, not the nouns of the ingredient list.*

*   **Advanced / Gourmet:** A recipe with three ingredients (like traditional French soufflé) can require extreme technical precision that keyword rules will completely miss.
*   **Easy:** A recipe with easy to find ingredients and easy to cook (does not need much prep time)
*   **One-Pot / One-Pan:** Rules looking for "pot" or "pan" fail when a recipe secretly requires multiple bowls for prep, marination, or separate resting stages.

---

## 5. Meal Role & Food Type
*What part of a meal an item plays. Keywords give a starting point, but the nutrient profile decides edge cases (e.g. "chicken" in a soup vs a plain breast).*

*   **Protein Centerpiece:** A meat, poultry, fish or seafood item a meal is built around. The product term names the cut, and the nutrients confirm it (high protein, almost no carbohydrate). Breaded, sauced or mixed dishes fall out on the carb / sodium profile rather than on a keyword list.
*   **Breakfast:** Items eaten in the morning: cereal, oatmeal, pancakes, waffles, eggs, bacon, breakfast sausage, muffins, bagels, yogurt. Eggs and bacon also appear in other meals, so this needs the combination of product, preparation (`scrambled`, `toasted`) and nutrients, not one keyword.
*   **Dessert / Sweet Treat:** Cakes, cookies, pies, puddings, ice cream and candy. Sugar content separates a real dessert from a savoury pie or an unsweetened "chocolate" baking ingredient.
*   **Drink:** Anything consumed as a beverage (coffee, tea, juice, soft drinks, milk, alcohol). The energy density and macro split are very different from solid foods, so the nutrient columns alone carry much of the signal. `beverage` is already detected as context in `recipe_labeling.ipynb`.
*   **Pantry Staple / Cooking Ingredient:** Items that are an input to cooking rather than something eaten as is: oils, spices, flour, dry rice / pasta / beans, nuts and seeds, syrups, vinegar. The `dry`, `unprepared`, `raw` adjectives plus a very high energy density (oils) or a starch-heavy profile make this learnable.
*   **Fresh Produce / Light & Fresh:** Raw or lightly prepared fruits and vegetables. Low energy density, fiber and low protein separate them from dried fruit, fried vegetables or vegetable dishes with sauce.

## 6. Preparation & Convenience
*How the item reaches the plate. This lives in the verbs (`frozen`, `canned`, `grilled`) that the embedding marks as a separate term type.*

*   **Ready-to-Eat / Convenience:** Frozen meals, canned goods, fast food, restaurant items and anything labelled `ready`, `instant` or `prepared`. The verb and adjective terms are the main signal. High sodium backs it up for processed items.
*   **Smoky / Grilled:** Food cooked with dry heat (grilled, broiled, roasted, smoked, barbecued, rotisserie). This is a flavour profile that comes from the cooking method, not the ingredient. It is complementary to `umami_rich` and covers many cooked meat rows that are `generic` today.

## 7. Texture & Satiety
*Sensory and "how filling" qualities that come from the macro balance rather than from a named ingredient.*

*   **Crispy / Crunchy:** Fried, toasted or baked-dry foods (chips, crackers, pretzels, fried chicken, nuts, granola). A frying or toasting verb plus a fat + carbohydrate-dense, low-moisture profile. A keyword rule on "fried" alone would miss crackers and nuts and wrongly include fried eggs.
*   **Hearty / Filling:** Foods with a high satiety value: lots of protein and fiber per calorie (lean meats, beans, lentils, eggs, whole grains). This is a nutrient ratio, so it is naturally learned from the nutrient components rather than from text.

### Notes for adding these to `recipe_labeling.ipynb`
- **Exclusivity:** `drink` should not co-occur with food-only labels such as `crispy_crunchy`, `protein_centerpiece` or `pantry_staple`, and `pantry_staple` should not co-occur with `ready_to_eat`.
- **Size:** `protein_centerpiece`, `hearty_filling` and `ready_to_eat` are large. They would reduce the `generic` share and the imbalance against it (currently `generic` vs `spicy` is about 60:1).
- **Existing labels:** `authentic_regional`, `holiday_celebration` and `game_day_snack` are no longer described above but are still produced by the notebook; drop them there too if they are retired.
- **Caveat:** these are still silver labels from rules over the same inputs, so model scores will measure how well the rules are recovered, not true label quality.
