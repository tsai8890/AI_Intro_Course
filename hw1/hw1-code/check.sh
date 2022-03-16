# Part1 A Star
echo "Part1 A Star" > result.txt
echo >> result.txt
echo "maps/single/bigMaze.txt:" >> result.txt
python3 hw1.py --method astar maps/single/bigMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/mediumMaze.txt:" >> result.txt
python3 hw1.py --method astar maps/single/mediumMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/openMaze.txt:" >> result.txt
python3 hw1.py --method astar maps/single/openMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/smallMaze.txt:" >> result.txt
python3 hw1.py --method astar maps/single/smallMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/tinyMaze.txt:" >> result.txt
python3 hw1.py --method astar maps/single/tinyMaze.txt >> result.txt
echo >> result.txt

# Part1 BFS 
echo "Part1 BFS" >> result.txt
echo >> result.txt
echo "maps/single/bigMaze.txt:" >> result.txt
python3 hw1.py --method bfs maps/single/bigMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/mediumMaze.txt:" >> result.txt
python3 hw1.py --method bfs maps/single/mediumMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/openMaze.txt:" >> result.txt
python3 hw1.py --method bfs maps/single/openMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/smallMaze.txt:" >> result.txt
python3 hw1.py --method bfs maps/single/smallMaze.txt >> result.txt
echo >> result.txt
echo "maps/single/tinyMaze.txt:" >> result.txt
python3 hw1.py --method bfs maps/single/tinyMaze.txt >> result.txt
echo >> result.txt

# Part2 A Star
echo "Part2 A Star" >> result.txt
echo >> result.txt
echo "maps/corner/bigCorners.txt:" >> result.txt
python3 hw1.py --method astar_corner maps/corner/bigCorners.txt >> result.txt
echo >> result.txt
echo "maps/corner/mediumCorners.txt:" >> result.txt
python3 hw1.py --method astar_corner maps/corner/mediumCorners.txt >> result.txt
echo >> result.txt
echo "maps/corner/tinyCorners.txt:" >> result.txt
python3 hw1.py --method astar_corner maps/corner/tinyCorners.txt >> result.txt
echo >> result.txt

# Part2 BFS
echo "Part2 BFS" >> result.txt
echo >> result.txt
echo "maps/corner/bigCorners.txt:" >> result.txt
python3 hw1.py --method bfs maps/corner/bigCorners.txt >> result.txt
echo >> result.txt
echo "maps/corner/mediumCorners.txt:" >> result.txt
python3 hw1.py --method bfs maps/corner/mediumCorners.txt >> result.txt
echo >> result.txt
echo "maps/corner/tinyCorners.txt:" >> result.txt
python3 hw1.py --method bfs maps/corner/tinyCorners.txt >> result.txt
echo >> result.txt

# Part3 A Star
echo "Part3 A Star" >> result.txt
echo >> result.txt
echo "maps/multi/greedySearch.txt:" >> result.txt
python3 hw1.py --method astar_multi maps/multi/greedySearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/mediumDottedMaze.txt:" >> result.txt
python3 hw1.py --method astar_multi maps/multi/mediumDottedMaze.txt >> result.txt
echo >> result.txt
echo "maps/multi/mediumSearch.txt:" >> result.txt
python3 hw1.py --method astar_multi maps/multi/mediumSearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/smallSearch.txt:" >> result.txt
python3 hw1.py --method astar_multi maps/multi/smallSearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/tinySearch.txt:" >> result.txt
python3 hw1.py --method astar_multi maps/multi/tinySearch.txt >> result.txt
echo >> result.txt

# Part3 BFS
echo "Part3 BFS" >> result.txt
echo >> result.txt
echo "maps/multi/greedySearch.txt:" >> result.txt
python3 hw1.py --method bfs maps/multi/greedySearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/mediumDottedMaze.txt:" >> result.txt
python3 hw1.py --method bfs maps/multi/mediumDottedMaze.txt >> result.txt
echo >> result.txt
echo "maps/multi/mediumSearch.txt:" >> result.txt
python3 hw1.py --method bfs maps/multi/mediumSearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/smallSearch.txt:" >> result.txt
python3 hw1.py --method bfs maps/multi/smallSearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/tinySearch.txt:" >> result.txt
python3 hw1.py --method bfs maps/multi/tinySearch.txt >> result.txt
echo >> result.txt

# Part4 A Star(Greedy)
echo "Part4 A Star" >> result.txt
echo >> result.txt
echo "maps/multi/mediumSearch_prev.txt:" >> result.txt
python3 hw1.py --method fast maps/multi/mediumSearch_prev.txt >> result.txt
echo >> result.txt
echo "maps/multi/bigSearch.txt:" >> result.txt
python3 hw1.py --method fast maps/multi/bigSearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/oddSearch.txt:" >> result.txt
python3 hw1.py --method fast maps/multi/oddSearch.txt >> result.txt
echo >> result.txt
echo "maps/multi/openSearch.txt:" >> result.txt
python3 hw1.py --method fast maps/multi/openSearch.txt >> result.txt
