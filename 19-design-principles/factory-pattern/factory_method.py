from abc import ABC, abstractmethod

from simple_factory import Character, WarriorCharacter, MageCharacter, CharacterFactory

class CharacterCreator(ABC):
    """Abstract creator class"""
    
    def create_party_member(self, name, level):
        """Factory method - subclasses decide which character to create"""
        character = self._create_character(name, level)
        self._customize_character(character)
        return character
    
    @abstractmethod
    def _create_character(self, name, level):
        """Factory method to be implemented by subclasses"""
        pass
    
    def _customize_character(self, character):
        """Hook for additional customization"""
        pass

class WarriorCreator(CharacterCreator):
    def _create_character(self, name, level):
        return WarriorCharacter(name, level)
    
    def _customize_character(self, character):
        # Warriors get bonus health
        character.health += 20

class MageCreator(CharacterCreator):
    def _create_character(self, name, level):
        return MageCharacter(name, level)
    
    def _customize_character(self, character):
        # Mages get bonus mana
        character.mana += 30


warrior_creator = WarriorCreator()
mage_creator = MageCreator()

warrior = warrior_creator.create_party_member("Arthur", 10)
mage = mage_creator.create_party_member("Merlin", 12)

print(f"Warrior: {warrior.name} - Health: {warrior.health}")
print(f"Mage: {mage.name} - Health: {mage.health}, Mana: {mage.mana}")

print("\n--- Dynamic Registration ---")
# Register a new character type at runtime
class PaladinCharacter(Character):
    def __init__(self, name, level):
        super().__init__(name, level)
        self.health = 120 + (level * 12)
        self.mana = 60 + (level * 8)
        self.damage = 22 + (level * 2)
        self.holy_power = 30 + (level * 3)
    
    def attack(self):
        return f"{self.name} strikes with holy power for {self.damage} damage!"

# Register new type without modifying factory
CharacterFactory.register_character_type("paladin", PaladinCharacter)

paladin = CharacterFactory.create_character("paladin", "Galahad", 9)
print(f"New character type: {paladin}")
print(paladin.attack())