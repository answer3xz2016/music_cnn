
#include <iostream>
#include <sstream>
#include <vector>
#include <queue>
#include <cstdlib>

#include <lemon/matching.h>
#include <lemon/smart_graph.h>
#include <lemon/concepts/graph.h>
#include <lemon/concepts/maps.h>
#include <lemon/lgf_reader.h>
#include <lemon/math.h>


using namespace std;
using namespace lemon;

GRAPH_TYPEDEFS(SmartGraph);


double checkWeightedMatching(const SmartGraph& graph,
			    const SmartGraph::EdgeMap<int>& weight,
			    const MaxWeightedMatching<SmartGraph>& mwm) {
  int pv = 0;
  for (SmartGraph::NodeIt n(graph); n != INVALID; ++n) {
    if (mwm.matching(n) != INVALID) {
      pv += weight[mwm.matching(n)];
    }
  }
  double maxWeight =  pv/2./100. ;
  return maxWeight;
}


int main(int argc, char** argv) {

  if(argc <= 1) return 1;
  
  const int numNodesParty1 = atoi(argv[1]);
  const int numNodesParty2 = atoi(argv[2]);

  if(argc != numNodesParty1*numNodesParty2+3) return 1;
  
  //cout<< numNodesParty1 << " - "<< numNodesParty2 << endl;

  string token = "@nodes\nlabel\n";
  for(int i =0 ; i< numNodesParty1 ; i++) token += "a" + std::to_string(i) + "\n";
  for(int i =0 ; i< numNodesParty2 ; i++) token += "b" + std::to_string(i) + "\n";
  token += "@edges\nweight\n";
  for(int i=0; i< numNodesParty1 ; i++){
    for(int j = 0; j < numNodesParty2 ; j++){
      token += "a" + std::to_string(i) + " " +
	"b" + std::to_string(j) + " " + argv[ i*numNodesParty2 + j + 3] + "\n" ;
    }
  }
  
  //cout << token << endl;
  

  SmartGraph graph;
  SmartGraph::EdgeMap<int> weight(graph);

  
  istringstream lgfs( token );

  graphReader(graph, lgfs).
    edgeMap("weight", weight).run();
  {
    MaxWeightedMatching<SmartGraph> mwm(graph, weight);
    mwm.run();
    cout<< checkWeightedMatching(graph, weight, mwm) << endl ;
  }
  
  return 0;


  
}
