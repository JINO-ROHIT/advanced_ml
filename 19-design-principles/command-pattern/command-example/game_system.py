from interface import GameCommand
from movement_command import MoveUpCommand, MoveDownCommand, MoveLeftCommand, MoveRightCommand, MoveCommand
from attack_command import AttackCommand
from player import GameActor

class GameSystem:
    def __init__(self, actor):
        self.actor = actor
        self.command_history = []
        self.current_position = -1
        self.input_buffer = []
        self.combos = {}
        self.recording_macro = False
        self.macro_commands = []
        

        self.key_bindings = {
            'w': MoveUpCommand(),
            's': MoveDownCommand(),
            'a': MoveLeftCommand(),
            'd': MoveRightCommand(),
            'space': AttackCommand(),
        }
    
    
    def execute_command(self, command: GameCommand):
        self.command_history = self.command_history[:self.current_position + 1]
        
        if command.can_execute(self.actor):
            command.execute(self.actor)
            self.command_history.append(command)
            self.current_position += 1
            
            if self.recording_macro:
                self.macro_commands.append(command)
    
    def handle_input(self, key):
        if key in self.key_bindings:
            command = self.key_bindings[key]
            self.execute_command(command)
        else:
            print(f"Unknown input: {key}")

    def undo(self):
        if self.current_position >= 0:
            command = self.command_history[self.current_position]
            command.undo(self.actor)
            self.current_position -= 1
        else:
            print("Nothing to undo")
    
    def redo(self):
        if self.current_position < len(self.command_history) - 1:
            self.current_position += 1
            command = self.command_history[self.current_position]
            command.execute(self.actor)
        else:
            print("Nothing to redo")
    
    def replay_actions():
        """you can implement later"""
        pass
    
    def show_history(self):
        print("\nAction History:")
        for i, command in enumerate(self.command_history):
            marker = "-> " if i == self.current_position else "   "
            print(f"{marker}{i}: {command.get_name()}")

print("=== AFTER: With Command Pattern ===")
player = GameActor("Hero")
game = GameSystem(player)

print("🎮 Starting game simulation...")
player.show_status()

print("\n--- Basic Actions ---")
inputs = ['d', 'd', 'w', 'space', 'h', 'f', 'space']
for key in inputs:
    game.handle_input(key)

print("\n--- Player Status ---")
player.show_status()

print("\n--- Testing Undo/Redo ---")
print("Undoing last 2 actions:")
game.undo()
game.undo()
player.show_status()

print("\nRedoing 1 action:")
game.redo()
player.show_status()

game.handle_input('space')  # attack

print("\nExecuting custom macro:")
game.handle_input('combo_power_move')

print("\n--- Action History ---")
game.show_history()

print(f"\nFinal Status:")
player.show_status()