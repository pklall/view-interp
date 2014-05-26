#pragma once

// OpenGM and Halide MUST be included before CImg, which includes
// X11 headers with conflicting/stupid definitions (i.e. `#define BOOL`).
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/simplediscretespace.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/sparsemarray.hxx"
#include "opengm/inference/messagepassing/messagepassing.hxx"
#include "opengm/inference/messagepassing/messagepassing_bp.hxx"
#include "opengm/inference/messagepassing/messagepassing_trbp.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/inference/mqpbo.hxx"
#include "opengm/inference/graphcut.hxx"
#include "opengm/inference/alphaexpansion.hxx"
#include "opengm/inference/alphabetaswap.hxx"
#include "opengm/inference/auxiliary/minstcutboost.hxx"

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
 *  UC - Unary cost class with function: float operator()(node_t, label_t)
 *  BC - Binary cost class with function: float operator()(node_t, node_t, label_t, label_t)
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

        /**
         * CandidateLabelFunc = function<label_t(node_t)>
         */
        template<class CandidateLabelFunc>
        void createExpandModel(
                const set<node_t>& nodes,
                const CandidateLabelFunc& labelFunc) {
            Space space(maxNodesPerExpansion, 2);

            model = GModel(space);

            int nodeIndex = 0;
            map<node_t, int> nodeMap;

            vector<node_t> neighbors;

            for (const node_t& node : nodes) {
                nodeMap[node] = nodeIndex;

                label_t newLabel = labelFunc(node);

                float costSame = unaryCost(node, (*labeling)[node]);
                float costDiff = unaryCost(node, newLabel);

                neighbors.clear();

                neighborGenerator(node, neighbors);

                for (const node_t& neighbor : neighbors) {
                    if (neighbor == node) {
                        continue;
                    }

                    label_t newNLabel = labelFunc(neighbor);

                    // If this neighbor is in the sub-graph to be modified
                    if (nodes.count(neighbor) > 0) {
                        // Only process pairs once (and only after both nodes
                        // have been added to nodeMap)
                        if (neighbor < node) {
                            size_t shape[] = {2, 2};

                            ExplicitFunction func(begin(shape), end(shape));

                            label_t oldLabel = (*labeling)[node];
                            label_t oldNLabel = (*labeling)[neighbor];

                            func(0, 0) = binaryCost(node, neighbor,
                                    oldLabel, oldNLabel);

                            func(0, 1) = binaryCost(node, neighbor,
                                    oldLabel, newNLabel);

                            func(1, 0) = binaryCost(node, neighbor,
                                    newLabel, oldNLabel);

                            func(1, 1) = binaryCost(node, neighbor,
                                    newLabel, newNLabel);

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

                        float sameLabelSameNeighborCost = binaryCost(node, neighbor,
                                (*labeling)[node], (*labeling)[neighbor]);

                        float sameLabelDiffNeighborCost = binaryCost(node, neighbor,
                                (*labeling)[node], newNLabel);

                        costSame += sameLabelSameNeighborCost;// min(sameLabelSameNeighborCost, sameLabelDiffNeighborCost);

                        float diffLabelSameNeighborCost = binaryCost(node, neighbor,
                                newLabel, (*labeling)[neighbor]);

                        float diffLabelDiffNeighborCost = binaryCost(node, neighbor,
                                newLabel, newNLabel);

                        costDiff += diffLabelSameNeighborCost; // min(diffLabelSameNeighborCost, diffLabelDiffNeighborCost);
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

        template<class CandidateLabelFunc>
        int tryExpand(
                const set<node_t>& nodes,
                CandidateLabelFunc& candidateGenerator) {
            createExpandModel(nodes, candidateGenerator);

            qpbo = unique_ptr<QPBO>(new QPBO(model));
            qpbo->infer();

            vector<bool> optimalVariables;
            vector<size_t> labels;

            qpbo->partialOptimality(optimalVariables);

            qpbo->arg(labels);
            
            int nodeIndex = 0;

            int numChanged = 0;

            for (const node_t& node : nodes) {
                if (optimalVariables[nodeIndex] && labels[nodeIndex] == 1) {
                    (*labeling)[node] = candidateGenerator(node);

                    numChanged++;
                }

                nodeIndex++;
            }

            return numChanged;
        }

        int tryExpand(
                const set<node_t>& nodes,
                label_t label) {
            struct LabelFunc {
                label_t label;

                label_t operator()(node_t node) const {
                    return this->label;
                }
            };

            LabelFunc func = {label};

            return tryExpand(nodes, func);
        }
};

