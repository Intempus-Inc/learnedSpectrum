import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import torch
from sklearn.manifold import TSNE


class TemporalUnderstandingVisualizer:
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_and_close(self, save_name: str) -> None:
        plt.savefig(self.save_dir / f'{save_name}.png')
        plt.close()

    def plot_physiological_dynamics(self, states_history: List[torch.Tensor],
                                  timestamps: List[float],
                                  save_name: str = 'physiological_dynamics'):
        plt.figure(figsize=(15, 10))

        if not states_history or all(s.numel() == 0 for s in states_history):
            plt.text(0.5, 0.5, 'No physiological states available',
                    ha='center', va='center')
            plt.title('Physiological State Dynamics (No Data)')
        else:
            valid_states = [s for s in states_history if s.numel() > 0]
            states_concat = torch.cat(valid_states, dim=0).cpu().numpy()
            states_2d = TSNE(n_components=2).fit_transform(states_concat)

            plt.scatter(states_2d[:, 0], states_2d[:, 1],
                       c=timestamps[:len(states_2d)], cmap='viridis', alpha=0.6)
            plt.colorbar(label='Time')
            plt.title('Physiological State Dynamics')
            plt.xlabel('TSNE Dimension 1')
            plt.ylabel('TSNE Dimension 2')

        self._save_and_close(save_name)

    def plot_causal_analysis(self, causal_data: Dict[str, Any],
                           save_name: str = 'causal_analysis'):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        G = nx.DiGraph()

        if isinstance(causal_data['edges'], list) and causal_data['edges']:
            for i in range(len(causal_data['edges']) - 1):
                G.add_edge(f'State_{i}', f'State_{i+1}')
        else:
            G.add_edge('Initial', 'Final')

        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=axes[0, 0], with_labels=True,
                node_color='lightblue', node_size=1500)
        axes[0, 0].set_title('Causal Relationship Graph')

        if isinstance(causal_data['temporal_dependencies'], torch.Tensor):
            temporal_deps = causal_data['temporal_dependencies'].cpu().numpy()
        else:
            temporal_deps = np.zeros((10, 10))

        sns.heatmap(temporal_deps, ax=axes[1, 0], cmap='coolwarm')
        axes[1, 0].set_title('Temporal Dependency Matrix')

        if isinstance(causal_data['treatment_effects'], list) and causal_data['treatment_effects']:
            treatment_effects = torch.stack(causal_data['treatment_effects']).cpu().numpy()
            sns.histplot(treatment_effects.flatten(), ax=axes[0, 1], bins=30)
        else:
            axes[0, 1].text(0.5, 0.5, 'No treatment effects data',
                           ha='center', va='center')
        axes[0, 1].set_title('Distribution of Treatment Effects')

        plt.tight_layout()
        self._save_and_close(save_name)

    def plot_temporal_understanding_metrics(self, metrics: Dict[str, Any],
                                         save_name: str = 'temporal_metrics'):
        fig, axes = plt.subplots(3, 2, figsize=(20, 25))

        axes[0, 0].plot(metrics['accuracy_history'], label='Accuracy')
        axes[0, 0].plot(metrics['loss_history'], label='Loss')
        axes[0, 0].set_title('Learning Progress')
        axes[0, 0].legend()

        sns.heatmap(metrics['ltc_time_constants'],
                   ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title('LTC Time Constants')

        axes[1, 0].plot(metrics['hjb_values'])
        axes[1, 0].set_title('HJB Value Function Evolution')

        sns.heatmap(metrics['q_values'], ax=axes[1, 1], cmap='RdYlBu')
        axes[1, 1].set_title('Q-Value Distribution')

        sns.heatmap(metrics['temporal_attention'],
                   ax=axes[2, 0], cmap='rocket')
        axes[2, 0].set_title('Temporal Attention Patterns')

        sns.heatmap(metrics['state_transitions'],
                   ax=axes[2, 1], cmap='YlOrRd')
        axes[2, 1].set_title('State Transition Probabilities')

        plt.tight_layout()
        self._save_and_close(save_name)

    def create_temporal_understanding_report(self, results: Dict[str, Any],
                                          save_name: str = 'temporal_report'):
        fig.add_subplot(3, 2, 5)
        self.plot_ltc_dynamics(results['ltc_analysis'])

        fig.add_subplot(3, 2, 6)
        self.plot_causal_inference(results['causal_analysis'])

        plt.tight_layout()
        self._save_and_close(save_name)

    def plot_ltc_dynamics(self, ltc_data: Dict[str, Any], save_name: str = 'ltc_dynamics'):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        sns.violinplot(data=ltc_data['time_constants'], ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of LTC Time Constants')

        trajectories = TSNE(n_components=2).fit_transform(ltc_data['hidden_states'])
        sns.scatterplot(
            x=trajectories[:, 0],
            y=trajectories[:, 1],
            hue=ltc_data['timestamps'],
            palette='viridis',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Hidden State Trajectories')

        sns.lineplot(
            data=ltc_data['adaptive_taus'],
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Adaptive Time Constant Evolution')

        sns.heatmap(
            ltc_data['state_transitions'],
            ax=axes[1, 1],
            cmap='coolwarm'
        )
        axes[1, 1].set_title('State Transition Probabilities')

        plt.tight_layout()
        self._save_and_close(save_name)

    def plot_causal_inference_results(self, causal_data: Dict[str, Any],
                                    save_name: str = 'causal_inference'):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        sns.boxplot(data=causal_data['intervention_effects'], ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Intervention Effects')

        G = nx.from_numpy_array(causal_data['adjacency_matrix'], create_using=nx.DiGraph)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=axes[0, 1],
               node_color='lightblue',
               node_size=1000,
               with_labels=True,
               arrows=True)
        axes[0, 1].set_title('Learned Causal Graph')

        sns.scatterplot(
            data=causal_data['treatment_effects'],
            x='propensity_score',
            y='effect_size',
            hue='subgroup',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Treatment Effect Heterogeneity')

        sns.heatmap(
            causal_data['temporal_strength'],
            ax=axes[1, 1],
            cmap='YlOrRd'
        )
        axes[1, 1].set_title('Temporal Causal Strength')

        plt.tight_layout()
        self._save_and_close(save_name)