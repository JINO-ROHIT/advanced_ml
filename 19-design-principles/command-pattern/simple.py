class SimplePlayer:
    def __init__(self, name):
        self.name = name
        self.x = 0
        self.y = 0
        self.health = 100
        self.mana = 50
        self.inventory = []
    
    def move_up(self):
        self.y += 1
        print(f"{self.name} moved up to ({self.x}, {self.y})")
    
    def move_down(self):
        self.y -= 1
        print(f"{self.name} moved down to ({self.x}, {self.y})")
    
    def move_left(self):
        self.x -= 1
        print(f"{self.name} moved left to ({self.x}, {self.y})")
    
    def move_right(self):
        self.x += 1
        print(f"{self.name} moved right to ({self.x}, {self.y})")
    
    def attack(self):
        if self.mana >= 10:
            self.mana -= 10
            print(f"{self.name} attacks! (Mana: {self.mana})")
        else:
            print(f"{self.name} not enough mana to attack!")
    
    def heal(self):
        if self.mana >= 15:
            self.mana -= 15
            self.health = min(100, self.health + 20)
            print(f"{self.name} heals! (health: {self.health}, mana: {self.mana})")
        else:
            print(f"{self.name} not enough mana to heal!")

class SimpleInputHandler:
    def __init__(self, player):
        self.player = player
    
    def handle_input(self, key):
        # Direct coupling - hard to extend or modify
        if key == 'w':
            self.player.move_up()
        elif key == 's':
            self.player.move_down()
        elif key == 'a':
            self.player.move_left()
        elif key == 'd':
            self.player.move_right()
        elif key == 'space':
            self.player.attack()
        elif key == 'h':
            self.player.heal()
        else:
            print(f"Unknown key: {key}")

player = SimplePlayer("Hero")
input_handler = SimpleInputHandler(player)

# Simulate some inputs
inputs = ['d', 'd', 'w', 'space', 'h', 'a']
for key in inputs:
    input_handler.handle_input(key)

print(f"Final position: ({player.x}, {player.y}), Health: {player.health}, Mana: {player.mana}")
print("but no replay, no undo, can't queue actions \n")


'''
you can probably store the state of the player and then try to undo from the previous action, it stores everything.
'''