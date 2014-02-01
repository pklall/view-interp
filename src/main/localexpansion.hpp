#pragma once

#include "common.h"

#include <vector>
#include <map>
#include <set>

using namespace std;

/**
 * Performs iterated graph-cuts over subsets of a graph.
 */
template<class N, class L>
class LocalExpansion {
    private:
        typedef N node_t;
        typedef L label_t;

        typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
        typedef opengm::ExplicitFunction<float, size_t, size_t> ExplicitFunction;

        typedef opengm::GraphicalModel<float, opengm::Adder, ExplicitFunction,
                Space> GModel;
        typedef opengm::external::QPBO<GModel> QPBO;

        GModel model;

        unique_ptr<QPBO> qpbo;

        vector<label_t>* labeling;

        /**
         * float binaryCost(
         *      node_t a,
         *      node_t b,
         *      label_t a_label,
         *      label_t b_label)
         *
         * Computes pair-wise cost between nodes a and b
         * when F_a = a_label and F_b = b_label for labeling F.
         */
        function<float(node_t, node_t, label_t, label_t)> binaryCost;

        /**
         * float unaryCost(
         *      node_t n,
         *      label_t n_label)
         *
         * Computes the cost of assigning node n the label n_label.
         */
        function<float(node_t, label_t)> unaryCost;

        /**
         * void neighborhoodGenerator(
         *      node_t n,
         *      vector<node_t>& neighbors)
         *
         * Creates a list of neighbors for the given node.
         */
        function<void(node_t, vector<node_t>&)> neighborhoodGenerator;

        void createExpandModel(
                const set<node_t>& nodes,
                label_t label) {
            Space space(nodes.size(), 2);

            model = GModel(space);

            int nodeIndex = 0;
            map<node_t, int> nodeMap;

            vector<node_t> neighbors;

            for (const node_t& node : nodes) {
                nodeMap[node] = nodeIndex;

                float costSame = unaryCost(node, (*labeling)[node]);
                float costDiff = unaryCost(node, label);

                neighborhoodGenerator(node, neighbors);

                for (const node_t& neighbor : neighbors) {
                    if (neighbor == node) {
                        continue;
                    }

                    // If this neighbor is in the sub-graph to be modified
                    if (nodes.count(neighbor) > 0) {
                        // Only process pairs once (and only after both nodes
                        // have been added to nodeMap)
                        if (neighbor < node) {
                            size_t shape[] = {2, 2};

                            ExplicitFunction func(begin(shape), end(shape));

                            label_t nLabel = 

                            func(0, 0) = binaryCost(node, neighbor,
                                    (*labeling)[node], (*labeling)[neighbor]);

                            func(0, 1) = binaryCost(node, neighbor,
                                    (*labeling)[node], label);

                            func(1, 0) = binaryCost(node, neighbor,
                                    label, (*labeling)[neighbor]);

                            func(1, 1) = binaryCost(node, neighbor,
                                    label, label);

                            GModel::FunctionIdentifier fid = model.addFunction(func);

                            size_t vars[] = {
                                (size_t) nodeMap[neighbor],
                                (size_t) nodeMap[node]};

                            model.addFactor(fid, begin(vars), end(vars));
                        }
                    } else {
                        // If this neighbor is static, we have to account
                        // for the added cost resulting from the pairwise
                        // potential by adding it to the current node's
                        // unary term.
                        costSame += binaryCost(node, neighbor,
                                (*labeling)[node], (*labeling)[neighbor]);

                        costDiff += binaryCost(node, neighbor,
                                label, (*labeling)[neighbor]);
                    }
                }

                size_t shape[] = {(size_t) 2};

                ExplicitFunction dataTerm(begin(shape), end(shape));

                dataTerm(0) = costSame;
                dataTerm(1) = costDiff;

                GModel::FunctionIdentifier fid = model.addFunction(dataTerm);

                size_t vars[] = {(size_t) nodeMap[node]};

                model.addFactor(fid, begin(vars), end(vars));

                nodeIndex++;
            }
        }

    public:
        LocalExpansion(
                vector<label_t>* _labeling,
                function<float(node_t, node_t, label_t, label_t)> _binaryCost,
                function<float(node_t, node_t)> _unaryCost,
                function<void(node_t, vector<node_t>&)> _neighborhoodGenerator) :
                    labeling(_labeling),
                    binaryCost(_binaryCost),
                    unaryCost(_unaryCost),
                    neighborhoodGenerator(_neighborhoodGenerator) {
        }

        void tryExpand(
                const set<node_t>& nodes,
                label_t label) {
            createExpandModel(nodes, label);

            qpbo = unique_ptr<QPBO>(new QPBO(model));
            qpbo->infer();

            vector<bool> optimalVariables;
            vector<size_t> labels;

            float partialOpt = qpbo->partialOptimality(optimalVariables);

            qpbo->arg(labels);
            
            int numChanged = 0;

            int nodeIndex = 0;

            for (const node_t& node : nodes) {
                if (optimalVariables[nodeIndex] && labels[nodeIndex] == 1) {
                    (*labeling)[node] = label;

                    numChanged++;
                }

                nodeIndex++;
            }
        }
};

