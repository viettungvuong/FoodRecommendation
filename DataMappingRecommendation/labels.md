# Complex Recipe Classification Labels (Non-Rule-Based)

This document covers recipe classification labels that **cannot be accurately solved using simple keyword filters or if-then rules**. These labels require machine learning models (like classifier chains) because they depend on contextual reasoning, ingredient ratios, technical nuance, and cultural context.

---

## 1. Authenticity, Origin & Fusion
*These labels require analyzing the specific combination and harmony of ingredients rather than just looking for geographic keywords.*

*   **Authentic Regional (e.g., Authentic Italian, Traditional Thai):** Rules fail because a dish can use regional ingredients (like soy sauce) but be a modern Western fusion dish. True authenticity relies on traditional ratios and core culinary pairings.
*   **Fusion Cuisine (e.g., Tex-Mex, Asian-Mexican Fusion):** Rules cannot easily distinguish where one cuisine ends and another begins based on raw text alone. 
*   **Street Food Style:** This relies on the context of portability, preparation speed, and cultural presentation rather than an ingredient checklist.

## 2. Occasion, Meal Course & Context
*These labels depend heavily on preparation complexity, presentation, and cultural habits.*

*   **Comfort Food:** Highly subjective and deeply tied to cultural context, textures, and rich flavor profiles rather than a fixed set of ingredients.
*   **Weeknight Dinner:** Requires evaluating prep time, cook time, tool usage, and overall cognitive load, which simple text filters cannot accurately measure.
*   **Holiday / Celebration Specific (e.g., Thanksgiving Side, Game Day Snack):** A dish becomes a "game day snack" based on serving style and portioning physics (finger foods), not just the presence of cheese or meat.
*   **Kid-Friendly:** Relies on the subtle absence of polarizing textures, visual appeal, and mild flavor profiles that rules struggle to evaluate.

## 3. Flavor Profiles & Sensory Qualities
*Flavors are chemical interactions. Rules fail because they cannot calculate how ingredients amplify, mask, or balance one another.*

*   **Spicy / Heat Level:** A single jalapeno in a 10-gallon pot is mild; three drops of extract in a small bowl is blazing. Rules struggle with scaling and ratios.
*   **Umami-Rich:** Requires detecting the synergistic effect of combining specific ingredients (like tomatoes + mushrooms + soy sauce) that trigger deep savory notes.
*   **Tangy / Zesty:** Relies on the acidic balance against fats and sugars. A rule tracking "lemon juice" will miss the balancing act of sugar that neutralizes sharpness.

## 4. Technical Complexity & Skill Level
*Complexity lives in the verbs of the instructions, not the nouns of the ingredient list.*

*   **Advanced / Gourmet:** A recipe with three ingredients (like traditional French soufflé) can require extreme technical precision that keyword rules will completely miss.
*   **One-Pot / One-Pan:** Rules looking for "pot" or "pan" fail when a recipe secretly requires multiple bowls for prep, marination, or separate resting stages.