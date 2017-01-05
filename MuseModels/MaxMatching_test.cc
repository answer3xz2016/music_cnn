
#include <iostream>
#include <sstream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <algorithm>    // copy
#include <iterator>     // ostream_operator

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


void split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}


std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}



int main(int argc, char** argv) {


  string fileName = "/a/muse_nebula_shared_data/google_news.csv_backup";
  string modelFileName = "/a/muse_nebula_shared_data/trained_model_Wikipedia.csv";

  vector < vector<string> > data;
  ifstream infile( fileName );

  while (infile)
    {
      string s;
      if (!getline( infile, s )) break;

      istringstream ss( s );
      vector <string> record;

      while (ss)
	{
	  string s;
	  if (!getline( ss, s, ',' )) break;
	  record.push_back( s );
	}

      data.push_back( split( record[3], ' ' ) );
    }
  
  cout << data.size() << endl;
  infile.close();
  
  ifstream infile2( modelFileName );
  std::map< string, vector<double> > myModel;
  
  while (infile2)
    {
      string s;
      if (!getline( infile2, s )) break;
      //cout << s << endl;
      istringstream ss( s );
      vector <double> record;
      string token;
      int mycount(0);
      while (ss)
	{
	  string s;
	  if (!getline( ss, s, ' ' )) break;
	  
	  if(mycount == 0) token = s;
	  else record.push_back( std::stod(s) );
	  mycount++;
	}
      
      if(record.size() != 400) continue;
		  
      myModel[token] = record;
      
    }

  cout<< "loaded model" << endl;
    

  

  /*
  
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
  */

  return 0;


  
}
