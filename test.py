from tree import MCTS
from gamenv import GameBoard

if __name__ == "__main__":
    gb = GameBoard()
    tree = MCTS(state=gb.board,parent=None,action=None,gb=gb,max_search_depth=10)
    node = tree.root
    gb.display_board()
    while not gb.game_over: 
        node = tree.mcts_search(root=node,iterations=300) #build tree
        print(f"Move Chosen: {node.action_to_str[node.action]} Score: {node.score}")
        gb.board = node.state
        gb.score = node.score
        print("Game board: ")
        gb.display_board()
        node.display_tree(max_depth=2)
        # cont = input("Press any key to continue.....")
