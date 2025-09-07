class GameActor:
    def __init__(self, name):
        self.name = name
        self.x = 0
        self.y = 0
        self.health = 100
        self.max_health = 100
        self.mana = 50
        self.max_mana = 50
        self.inventory = []
        self.level = 1
        self.experience = 0
    
    def move_to(self, x, y):
        old_pos = (self.x, self.y)
        self.x, self.y = x, y
        return old_pos
    
    def change_health(self, amount):
        old_health = self.health
        self.health = max(0, min(self.max_health, self.health + amount))
        return old_health
    
    def change_mana(self, amount):
        old_mana = self.mana
        self.mana = max(0, min(self.max_mana, self.mana + amount))
        return old_mana
    
    def add_experience(self, exp):
        old_exp = self.experience
        self.experience += exp
        if self.experience >= 100:  # Level up
            self.level += 1
            self.experience = 0
            self.max_health += 10
            self.max_mana += 5
            print(f"🎉 {self.name} leveled up to level {self.level}!")
        return old_exp
    
    def show_status(self):
        print(f"{self.name} - Position: ({self.x}, {self.y}), Health: {self.health}/{self.max_health}, "
              f"Mana: {self.mana}/{self.max_mana}, Level: {self.level}, XP: {self.experience}")