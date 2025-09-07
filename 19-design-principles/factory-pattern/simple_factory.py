from abc import ABC, abstractmethod

class Character(ABC):
    def __init__(self, name, level):
        self.name = name
        self.level = level
    
    @abstractmethod
    def attack(self):
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.name} (Level {self.level})"

class WarriorCharacter(Character):
    def __init__(self, name, level):
        super().__init__(name, level)
        self.health = 100 + (level * 10)
        self.damage = 20 + (level * 2)
        self.weapon = "Sword"
        self.armor = "Chain Mail"
    
    def attack(self):
        return f"{self.name} swings {self.weapon} for {self.damage} damage!"

class MageCharacter(Character):
    def __init__(self, name, level):
        super().__init__(name, level)
        self.health = 60 + (level * 5)
        self.mana = 100 + (level * 15)
        self.damage = 30 + (level * 3)
        self.spell = "Fireball"
    
    def attack(self):
        return f"{self.name} casts {self.spell} for {self.damage} damage!"

class ArcherCharacter(Character):
    def __init__(self, name, level):
        super().__init__(name, level)
        self.health = 80 + (level * 7)
        self.damage = 25 + (level * 2.5)
        self.weapon = "Bow"
        self.arrows = 50 + (level * 5)
    
    def attack(self):
        return f"{self.name} shoots {self.weapon} for {self.damage} damage!"

class RogueCharacter(Character):
    def __init__(self, name, level):
        super().__init__(name, level)
        self.health = 70 + (level * 6)
        self.damage = 18 + (level * 3)
        self.stealth = 50 + (level * 5)
        self.weapon = "Dagger"
    
    def attack(self):
        return f"{self.name} backstabs with {self.weapon} for {self.damage} damage!"

class CharacterFactory:
    # registry of available character types
    _character_types = {
        "warrior": WarriorCharacter,
        "mage": MageCharacter,
        "archer": ArcherCharacter,
        "rogue": RogueCharacter
    }
    
    @classmethod
    def create_character(cls, char_type, name, level):
        """Create a character of the specified type"""
        if char_type not in cls._character_types:
            available = ", ".join(cls._character_types.keys())
            raise ValueError(f"Unknown character type: {char_type}. Available: {available}")
        
        character_class = cls._character_types[char_type]
        return character_class(name, level)
    
    @classmethod
    def register_character_type(cls, type_name, character_class):
        """Register a new character type"""
        cls._character_types[type_name] = character_class
    
    @classmethod
    def get_available_types(cls):
        """Get list of available character types"""
        return list(cls._character_types.keys())

characters = []
character_configs = [
    ("warrior", "Conan", 5),
    ("mage", "Gandalf", 8), 
    ("archer", "Legolas", 6),
    ("rogue", "Robin", 7)  # New type easily added
]

for char_type, name, level in character_configs:
    character = CharacterFactory.create_character(char_type, name, level)
    characters.append(character)
    print(f"Created: {character}")

print(f"Available character types: {CharacterFactory.get_available_types()}")