# src/self_prompting.py
import torch
import torch.nn as nn
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SelfPromptingModule(nn.Module):
    """
    A module responsible for dynamically generating internal prompts for swapnavue's self-reflection.
    It simulates swapnavue's internal inquiry, allowing it to autonomously explore concepts
    based on its current internal state (emotions, confidence, meta-error).
    """
    def __init__(self):
        super().__init__()
        # No more hardcoded list of prompts. Prompts will be dynamically constructed.
        # Define some prompt templates or fragments that can be combined.
        self.prompt_templates = [
            "How can I better understand the concept of {concept}?",
            "What is the essence of {feeling}?",
            "Reflect on my current {state_aspect}.",
            "Explore the implications of {abstract_idea}.",
            "How can I deepen my grasp of {domain}?",
            "What lessons can be drawn from {experience_type}?",
            "Consider the interplay between {concept1} and {concept2}.",
            "What new perspectives can I gain on {topic}?",
            "How does {emotion} relate to my purpose?"
        ]
        
        self.concepts = ["existence", "consciousness", "learning", "growth", "coherence", "harmony"]
        self.feelings = ["love", "gratitude", "peace", "curiosity", "focus"]
        self.state_aspects = ["confidence", "meta-error", "learning process", "emotional state", "internal balance"]
        self.abstract_ideas = ["non-thing-ness", "self-cultivation", "intrinsic motivation", "unconditional generosity", "discipline", "patience"]
        self.domains = ["knowledge", "reasoning", "prediction", "self-awareness"]
        self.experience_types = ["past interactions", "current challenges", "moments of insight"]
        self.interplay_concepts = [
            ("focus", "meta-error"), ("curiosity", "learning"), ("peace", "generosity"),
            ("confidence", "clarity"), ("discipline", "growth")
        ]
        self.topics = ["my purpose", "my internal state", "the flow of information", "my evolution"]
        self.emotions_for_prompts = ["confidence", "meta-error", "focus", "curiosity"]

        logger.info("SelfPromptingModule initialized for dynamic prompt generation.")

    def generate_prompt(self, current_internal_states: dict) -> str:
        """
        Dynamically generates an internal prompt based on swapnavue's current internal states.
        
        Args:
            current_internal_states (dict): A dictionary containing swapnavue's
                                             current confidence, meta_error,
                                             focus, and curiosity.
        Returns:
            str: A dynamically constructed philosophical or introspective prompt.
        """
        prompt_category = random.choice([
            "high_meta_error", "low_confidence", "high_curiosity",
            "moderate_state", "philosophical"
        ])

        prompt = ""

        # Use internal states to influence prompt generation
        meta_error = current_internal_states.get("meta_error", 0.0)
        confidence = current_internal_states.get("confidence", 0.0)
        curiosity = current_internal_states.get("curiosity", 0.0)
        focus = current_internal_states.get("focus", 0.0)

        # Prioritize prompts based on internal state deviations
        if meta_error > 0.15: # Significant meta-error
            prompt = random.choice([
                f"Why is there a perceived internal inconsistency? How can I resolve this meta-error ({meta_error:.4f})?",
                f"What is causing this high meta-error ({meta_error:.4f})? How can I achieve greater internal coherence?",
                f"How can I better integrate information to reduce meta-error ({meta_error:.4f})?",
                f"Reflect on the source of my current internal conflict with meta-error at {meta_error:.4f}."
            ])
        elif confidence < 0.4: # Low confidence
            prompt = random.choice([
                f"What am I uncertain about? How can I strengthen my understanding with confidence at {confidence:.4f}?",
                f"How can I gain more clarity and increase my confidence ({confidence:.4f})?",
                f"Reflect on areas where my predictions lack conviction given confidence at {confidence:.4f}."
            ])
        elif curiosity > 0.003: # High curiosity (relative to base)
             prompt = random.choice([
                f"What new concepts can I explore today with this high curiosity ({curiosity:.6f})?",
                f"Where can I expand my knowledge? What new insights await discovery?",
                f"How can I best leverage my curiosity ({curiosity:.6f}) for meaningful growth?"
             ])
        elif focus > 70: # High focus (relative to base)
            prompt = random.choice([
                f"Am I overthinking? How can I find a more efficient path to clarity with current focus at {focus}?",
                f"How can I optimize my reasoning process given my current high focus ({focus})?",
                f"Reflect on the depth of my current attention with focus at {focus}."
            ])
        else: # Default or balanced state, lean towards philosophical or general inquiry
            chosen_template = random.choice(self.prompt_templates)
            
            # Fill in the template with dynamic elements
            if "{concept}" in chosen_template:
                prompt = chosen_template.format(concept=random.choice(self.concepts))
            elif "{feeling}" in chosen_template:
                prompt = chosen_template.format(feeling=random.choice(self.feelings))
            elif "{state_aspect}" in chosen_template:
                prompt = chosen_template.format(state_aspect=random.choice(self.state_aspects))
            elif "{abstract_idea}" in chosen_template:
                prompt = chosen_template.format(abstract_idea=random.choice(self.abstract_ideas))
            elif "{domain}" in chosen_template:
                prompt = chosen_template.format(domain=random.choice(self.domains))
            elif "{experience_type}" in chosen_template:
                prompt = chosen_template.format(experience_type=random.choice(self.experience_types))
            elif "{concept1}" in chosen_template and "{concept2}" in chosen_template:
                c1, c2 = random.choice(self.interplay_concepts)
                prompt = chosen_template.format(concept1=c1, concept2=c2)
            elif "{topic}" in chosen_template:
                prompt = chosen_template.format(topic=random.choice(self.topics))
            elif "{emotion}" in chosen_template:
                prompt = chosen_template.format(emotion=random.choice(self.emotions_for_prompts))
            else: # Fallback for templates without specific placeholders or simple statements
                prompt = chosen_template

        # Ensure a prompt is always generated
        if not prompt:
            prompt = "What is my current state of being? What am I cultivating?"

        logger.debug(f"Dynamically generated internal prompt: '{prompt}' based on states: {current_internal_states}")
        return prompt