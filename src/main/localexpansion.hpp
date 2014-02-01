#pragma once

#include "common.h"

#include <vector>
#include <map>
#include <set>

using namespace std;

/**
 * Performs iterated graph-cuts over subsets of a graph.
 *
 * Type parameters:
 *  N - node type
 *  L - label type
 *  UC - Unary cost class with function: float operator()(node_t, node_t, label_t, label_t)
 *  BC - Binary cost class with function: float operator()(node_t, label_t)
 */
template<class N, class L, class UC, class BC>
class LocalExpansion {
    private:
        typedef N node_t;
        typedef L label_t;
        typedef UC unary_cost_t;
        typedef BC binary_cost_t;

        typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
        typedef opengm::ExplicitFunction<float, size_t, size_t> ExplicitFunction;

        typedef opengm::GraphicalModel<float, opengm::Adder, ExplicitFunction,
                Space> GModel;

        typedef opengm::external::QPBO<GModel> QPBO;

        size_t maxNodesPerExpansion;

        GModel model;

        unique_ptr<QPBO> qpbo;

        vector<label_t>* labeling;

        /**
         * float unaryCost(
         *      node_t n,
         *      label_t n_label)
         *
         * Computes the cost of assigning node n the label n_label.
         */
        unary_cost_t unaryCost;

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
        binary_cost_t binaryCost;

        /**
         * void neighborGenerator(
         *      node_t n,
         *      vector<node_t>& neighbors)
         *
         * Creates a list of neighbors for the given node.
         */
        function<void(node_t, vector<node_t>&)> neighborGenerator;

        void createExpandModel(
                const set<node_t>& nodes,
                label_t label) {
            Space space(maxNodesPerExpansion, 2);

            model = GModel(space);

            int nodeIndex = 0;
            map<node_t, int> nodeMap;

            vector<node_t> neighbors;

            for (const node_t& node : nodes) {
                nodeMap[node] = nodeIndex;

                float costSame = unaryCost(node, (*labeling)[node]);
                float costDiff = unaryCost(node, label);

                neighbors.clear();

                neighborGenerator(node, neighbors);

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
                size_t _maxNodesPerExpansion,
                vector<label_t>* _labeling,
                unary_cost_t _unaryCost,
                binary_cost_t _binaryCost,
                function<void(node_t, vector<node_t>&)> _neighborGenerator) :
                    maxNodesPerExpansion(_maxNodesPerExpansion),
                    labeling(_labeling),
                    unaryCost(_unaryCost),
                    binaryCost(_binaryCost),
                    neighborGenerator(_neighborGenerator) {
        }

        void tryExpand(
                const set<node_t>& nodes,
                label_t label) {
            createExpandModel(nodes, label);

            qpbo = unique_ptr<QPBO>(new QPBO(model));
            qpbo->infer();

            vector<bool> optimalVariables;
            vector<size_t> labels;

            qpbo->partialOptimality(optimalVariables);

            qpbo->arg(labels);
            
            int nodeIndex = 0;

            for (const node_t& node : nodes) {
                if (optimalVariables[nodeIndex] && labels[nodeIndex] == 1) {
                    (*labeling)[node] = label;
                }

                nodeIndex++;
            }
        }
};

