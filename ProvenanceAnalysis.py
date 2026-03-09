import os 
import itertools
import networkx as nx
import PatternMatching as pm

class ProvenanceAnalyzer:
    def __init__(self, similaritythreshold = 25.0):
        self.similaritythreshold = similaritythreshold
        self.graph = nx.DiGraph()

    def comparePair(self, path1, path2, featurecache):
        img1, kp1, desc1, px1 = featurecache[path1]
        img2, kp2, desc2, px2 = featurecache[path2]
        evidence1_2 = pm.compareFromFeatures(img1, kp1, desc1, px1, img2, kp2, desc2, px2)
        evidence2_1 = pm.compareFromFeatures(img2, kp2, desc2, px2, img1, kp1, desc1, px1)
        return evidence1_2, evidence2_1

    def buildSimilarityMatrix(self, imgpaths):
        similarities = {}

        featurecache = {}
        for path in imgpaths:
            img, pixels = pm.loadImage(path)
            kp, desc = pm.extractSIFT(img)
            featurecache[path] = (img, kp, desc, pixels)
            print(f"Loaded {path}: {pixels} pixels, {len(kp)} keypoints")
    
        for img1, img2 in itertools.combinations(imgpaths, 2):
            print(f"Comparing {img1} and {img2}")
            evidence1_2, evidence2_1 = self.comparePair(img1, img2, featurecache)
            similarities[(img1, img2)] = (evidence1_2, evidence2_1)
    
        return similarities
    
    def buildProvenanceGraph(self, similarities):
        graph = nx.DiGraph()
        
        for (imgA, imgB), (evidenceAB, evidenceBA) in similarities.items():
            bestscore = max(evidenceAB.score, evidenceBA.score)
            
            if bestscore >= self.similaritythreshold:
                src, dst = self.inferDirection(imgA, imgB, evidenceAB, evidenceBA)
                graph.add_edge(src, dst, weight=bestscore)
                
        self.graph = graph
        return graph

    def optimiseGraph(self):
        finalgraph = nx.DiGraph()
        
        for component in nx.weakly_connected_components(self.graph):
            subgraph = self.graph.subgraph(component).copy()
            
            if len(subgraph.nodes) == 1:
                finalgraph.add_node(list(subgraph.nodes)[0])
                continue
            
            mst = nx.maximum_spanning_arborescence(subgraph)
            finalgraph = nx.compose(finalgraph, mst)
        self.graph = finalgraph

    def analyzeProvenance(self, folderpath):
        
        imagepaths = [os.path.join(folderpath, f)
                    for f in os.listdir(folderpath)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        
        similarities = self.buildSimilarityMatrix(imagepaths)
        self.buildProvenanceGraph(similarities)
        self.optimiseGraph()
    
        return self.graph
    
    def inferDirection(self, imgA, imgB, evidenceAB, evidenceBA):
        pixelsA = evidenceAB.img1pixels
        pixelsB = evidenceAB.img2pixels

        if pixelsA > pixelsB * 1.05:
            return imgA, imgB
        elif pixelsB > pixelsA * 1.05:
            return imgB, imgA

        if evidenceAB.score >= evidenceBA.score:
            return imgB, imgA
        else:
            return imgA, imgB
    
    def rootCandidates(self):
        roots = [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]
        return roots


if __name__ == "__main__":
    datasetfolder = r"C:\Users\User\Documents\GitHub\FYP---Digital-Forensics\TestPath"

    analyser = ProvenanceAnalyzer(similaritythreshold = 25.0)
    provenancegraph = analyser.analyzeProvenance(datasetfolder)

    print("\nInferred Provenance Graph:")
    for u, v, data in provenancegraph.edges(data = True):
        print(f"{u} -> {v} (similarity: {data['weight']:.2f}%)")

    roots = analyser.rootCandidates()
    print("\nMost likely root candidate(s):")
    for root in roots:
        print(root)