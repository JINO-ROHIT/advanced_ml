class Warrior:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.health = 100 + (level * 10)
        self.damage = 20 + (level * 2)
        self.weapon = "Sword"
        self.armor = "Chain Mail"
    
    def attack(self):
        return f"{self.name} swings {self.weapon} for {self.damage} damage!"

class Mage:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.health = 60 + (level * 5)
        self.mana = 100 + (level * 15)
        self.damage = 30 + (level * 3)
        self.spell = "Fireball"
        self.staff = "Magic Staff"
    
    def attack(self):
        return f"{self.name} casts {self.spell} for {self.damage} damage!"

class Archer:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.health = 80 + (level * 7)
        self.damage = 25 + (level * 2.5)
        self.weapon = "Bow"
        self.arrows = 50 + (level * 5)
    
    def attack(self):
        return f"{self.name} shoots {self.weapon} for {self.damage} damage!"

class BadCharacterCreator:
    def create_party(self, party_config):
        party = []
        
        for char_info in party_config:
            char_type = char_info["type"]
            name = char_info["name"]
            level = char_info["level"]
            
            # repetitive and hard to maintain
            if char_type == "warrior":
                character = Warrior(name, level)
            elif char_type == "mage":
                character = Mage(name, level)
            elif char_type == "archer":
                character = Archer(name, level)
            else:
                raise ValueError(f"Unknown character type: {char_type}")
            
            party.append(character)
        
        return party

bad_creator = BadCharacterCreator()
party_config = [
    {"type": "warrior", "name": "Conan", "level": 5},
    {"type": "mage", "name": "Gandalf", "level": 8},
    {"type": "archer", "name": "Legolas", "level": 6}
]

party = bad_creator.create_party(party_config)
for character in party:
    print(f"{character.name} - Health: {character.health}")