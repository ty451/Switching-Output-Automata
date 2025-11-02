import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass

class Container:
    """模拟MATLAB的containers.Map类"""
    def __init__(self):
        self._map = {}
    
    def __setitem__(self, key, value):
        self._map[key] = value
    
    def __getitem__(self, key):
        return self._map.get(key, 1)  # 默认返回1
    
    def keys(self):
        return list(self._map.keys())
    
    def values(self):
        return list(self._map.values())
    
    def items(self):
        return self._map.items()
        
    def __contains__(self, key):
        return key in self._map

@dataclass
class AutomatonResult:
    """存储自动机结果的数据类"""
    Q: List[Any]  # 状态可以是字符串或列表
    Ye: List[str]
    Delta: List[Any]  # 转换可以有不同格式
    q0: Any  # 初始状态可以是字符串或列表
    observation: Dict[str, str]
    Y_f: List[str] = None
    Delta_o: List[str] = None

def sigma(x: str, B: List[List[str]]) -> List[str]:
    """寻找从状态x出发的所有可能转换目标状态"""
    return [x_prime for x_current, x_prime in B if x_current == x]

def D_y(q: str, y: str, Ge: Any, observation: Dict) -> List[str]:
    """找到从状态q经过输出y可达的所有状态"""
    D_y_q = []
    for from_state, event, target_state in Ge.Delta:
        if from_state == q and observation[event] == y:
            if target_state not in D_y_q:
                D_y_q.append(target_state)
    return D_y_q

def D_epsilon(q: str, Ge: Any, observation: Dict) -> List[str]:
    """找到从状态q经过epsilon转换可达的所有状态"""
    D_epsilon_q = [q]
    queue = [q]
    
    while queue:
        current_state = queue.pop(0)
        for from_state, event, target_state in Ge.Delta:
            if from_state == current_state and observation[event] == 'epsilon':
                if target_state not in D_epsilon_q:
                    D_epsilon_q.append(target_state)
                    queue.append(target_state)
    return D_epsilon_q

def generate_global_states(h: Container) -> List[str]:
    """生成所有可能的全局状态"""
    return [f"({key},{value})" for key in h.keys() for value in h[key]]

def get_upper_limit(G_tf: Any, global_states: List[str]) -> Container:
    """计算每个全局状态的时间上限"""
    max_mapping = Container()
    
    # 设置所有状态的默认值
    for state in global_states:
        max_mapping[state] = 1
    
    # 更新故障区间的值
    for key, intervals in G_tf.FaultOccurrenceIntervals.items():
        state = key.split(')(')[0] + ')'
        if isinstance(intervals[0], list):
            max_value = max(interval[1] for interval in intervals)
        else:
            max_value = intervals[1]
        max_mapping[state] = max_value
    
    return max_mapping

def construct_timed_fault_evolution_automaton(G_tf):
    """构造时间故障演化自动机"""
    global_states = generate_global_states(G_tf.h)
    max_mapping = get_upper_limit(G_tf, global_states)
    
    Yr = [str(y) for y in G_tf.Y]
    Ye = Yr + ['delta']
    Yf = []
    q0 = f"({G_tf.x0},{G_tf.y0})0"
    Qnew = [q0]
    Q = []
    Delta = []
    observation = Container()
    
    while Qnew:
        q = Qnew.pop(0)
        try:
            state_part = q[:q.find(')')+1]
            time_part = q[q.find(')')+1:]
            x = state_part[1:state_part.find(',')]
            y = state_part[state_part.find(',')+1:state_part.find(')')]
            x_j = time_part
            x_g = state_part
            
            num = int(x_j)
            max_limit = max_mapping[x_g]
            j_set = range(int(max_limit))
            
            if num in j_set:
                q_bar = f"{x_g}{num+1}"
                Delta.append((q, 'delta', q_bar))
                observation['delta'] = 'delta'
                if q_bar not in Q + Qnew:
                    Qnew.append(q_bar)
            elif num == int(max_limit):
                Delta.append((q, 'delta', q))
                observation['delta'] = 'delta'
            
            if num > 0:
                # 常规转换
                for x_bar in sigma(x, G_tf.B):
                    if int(y) in G_tf.h[x_bar]:
                        q_bar = f"({x_bar},{y})0"
                        Delta.append((q, 'epsilon_r', q_bar))
                        observation['epsilon_r'] = 'epsilon'
                        if q_bar not in Q + Qnew:
                            Qnew.append(q_bar)
                        if 'epsilon_r' not in Yr:
                            Yr.append('epsilon_r')
                    
                    for y_bar in set(G_tf.h[x_bar]) - {int(y)}:
                        q_bar = f"({x_bar},{y_bar})0"
                        Delta.append((q, str(y_bar), q_bar))
                        observation[str(y_bar)] = str(y_bar)
                        if q_bar not in Q + Qnew:
                            Qnew.append(q_bar)
                
                # 输出变化
                for y_bar in set(G_tf.h[x]) - {int(y)}:
                    q_bar = f"({x},{y_bar})0"
                    Delta.append((q, str(y_bar), q_bar))
                    observation[str(y_bar)] = str(y_bar)
                    if q_bar not in Q + Qnew:
                        Qnew.append(q_bar)
                
                # 故障转换
                for x_bar in sigma(x, G_tf.B_f):
                    transition_key = f"{x_g}({x},{x_bar})"
                    if transition_key in G_tf.FaultOccurrenceIntervals:
                        intervals = G_tf.FaultOccurrenceIntervals[transition_key]
                        is_in_interval = False
                        
                        if isinstance(intervals[0], list):
                            for interval in intervals:
                                if interval[0] <= num < interval[1]:
                                    is_in_interval = True
                                    break
                        else:
                            if intervals[0] <= num < intervals[1]:
                                is_in_interval = True
                        
                        if is_in_interval:
                            if int(y) in G_tf.h[x_bar]:
                                q_bar = f"({x_bar},{y})0"
                                Delta.append((q, 'epsilon_f', q_bar))
                                observation['epsilon_f'] = 'epsilon'
                                if q_bar not in Q + Qnew:
                                    Qnew.append(q_bar)
                                if 'epsilon_f' not in Yf:
                                    Yf.append('epsilon_f')
                            
                            for y_bar in set(G_tf.h[x_bar]) - {int(y)}:
                                q_bar = f"({x_bar},{y_bar})0"
                                event = f"{y_bar}_f"
                                Delta.append((q, event, q_bar))
                                observation[event] = str(y_bar)
                                if q_bar not in Q + Qnew:
                                    Qnew.append(q_bar)
                                if event not in Yf:
                                    Yf.append(event)
        
        except (ValueError, KeyError) as e:
            print(f"Warning: Error processing state {q}: {e}")
            continue
        
        Q.append(q)
    
    Ye = list(set(Ye + Yr + Yf))
    
    return AutomatonResult(
        Q=Q,
        Ye=Ye,
        Delta=Delta,
        q0=q0,
        Y_f=Yf,
        observation=observation
    )

def construct_fault_monitor(G_tfe: Any) -> AutomatonResult:
    """构建故障监视器"""
    x0 = 'N'
    Delta = []
    Xnew = [x0]
    X = []
    Y = G_tfe.Ye
    Y_f = G_tfe.Y_f if G_tfe.Y_f else []
    
    while Xnew:
        x = Xnew.pop(0)
        if x == 'N':
            for s in Y:
                if s not in Y_f:
                    Delta.append((x, str(s), x))
                else:
                    x_bar = 'F'
                    Delta.append((x, str(s), x_bar))
                    if x_bar not in X + Xnew:
                        Xnew.append(x_bar)
        else:
            for s in Y:
                Delta.append((x, str(s), x))
        
        X.append(x)
    
    return AutomatonResult(
        Q=X,
        Ye=Y,
        Delta=Delta,
        q0=x0,
        observation={str(y): str(y) for y in Y}
    )

def concurrent_composition(G_tfe, G_M):
    """构造并发组合"""
    Q_CC = []
    E_CC = G_tfe.Ye
    Delta_CC = []
    q0_CC = [G_tfe.q0, G_M.q0]
    
    Q = [q0_CC]
    
    while Q:
        q_CC = Q.pop(0)
        
        # 检查状态是否已经在Q_CC中，如果是则跳过
        if q_CC in Q_CC:
            continue
            
        for s in E_CC:
            s_transitions = [(i, t) for i, t in enumerate(G_tfe.Delta) 
                           if t[0] == q_CC[0] and t[1] == str(s)]
            Q_NFA_prime = [t[2] for _, t in s_transitions]
            
            s_transitions_1 = [(i, t) for i, t in enumerate(G_M.Delta) 
                             if t[0] == q_CC[1] and t[1] == str(s)]
            q_DFA_prime = [t[2] for _, t in s_transitions_1]
            
            if q_DFA_prime:
                q_DFA_prime1 = q_DFA_prime[0]
                
                for q_NFA_prime in Q_NFA_prime:
                    new_state = [q_NFA_prime, q_DFA_prime1]
                    
                    # 确保新状态不在Q_CC和Q中
                    if new_state not in Q_CC and new_state not in Q:
                        Q.append(new_state)
                    
                    Delta_CC.append((q_CC, s, new_state))
        
        Q_CC.append(q_CC)
    
    return AutomatonResult(
        Q=Q_CC,
        Ye=E_CC,
        Delta=Delta_CC,
        q0=q0_CC,
        observation=G_tfe.observation
    )

def construct_observer(Ge: Any) -> AutomatonResult:
    """构建观察器"""
    z_0 = D_epsilon(Ge.q0, Ge, Ge.observation)
    Z = [z_0]
    Z_prime = []
    Delta = []
    
    while Z:
        z = Z.pop(0)
        
        for y in Ge.Ye:
            alpha = []
            for q in z:
                xx = D_y(q, y, Ge, Ge.observation)
                for x in xx:
                    if x not in alpha:
                        alpha.append(x)
            
            beta = []
            for qq in alpha:
                xxx = D_epsilon(qq, Ge, Ge.observation)
                for x in xxx:
                    if x not in beta:
                        beta.append(x)
            
            z_bar = beta
            
            if z_bar:
                z_str = ','.join(sorted(z))
                z_bar_str = ','.join(sorted(z_bar))
                transition = (z_str, y, z_bar_str)
                
                if transition not in Delta:
                    Delta.append(transition)
                
                if z_bar not in Z + Z_prime:
                    Z.append(z_bar)
        
        Z_prime.append(z)
    
    return AutomatonResult(
        Q=Z + Z_prime,
        Ye=Ge.Ye,
        Delta=Delta,
        q0=z_0,
        observation=Ge.observation
    )

def merge_cc_states(cc: Any) -> AutomatonResult:
    """合并并发组合状态"""
    Q_new = []
    for state_pair in cc.Q:
        merged_state = state_pair[0] + state_pair[1]
        # 确保不添加重复的状态
        if merged_state not in Q_new:
            Q_new.append(merged_state)
    
    Delta_new = []
    for from_state, event, to_state in cc.Delta:
        merged_from = from_state[0] + from_state[1]
        merged_to = to_state[0] + to_state[1]
        transition = (merged_from, event, merged_to)
        # 确保不添加重复的转换
        if transition not in Delta_new:
            Delta_new.append(transition)
    
    return AutomatonResult(
        Q=Q_new,
        Ye=cc.Ye,
        Delta=Delta_new,
        q0=cc.q0[0] + cc.q0[1],
        observation=cc.observation
    )

def save_to_excel(G_tfe: AutomatonResult, G_M: AutomatonResult, 
                 cc: AutomatonResult, result: AutomatonResult, 
                 Gobs: AutomatonResult, filename: str = 'results.xlsx'):
    """将结果保存到Excel"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # G_tfe
        # 确保所有列的长度一致
        max_len = max(len(G_tfe.Q), len(G_tfe.Ye), 1)
        states = G_tfe.Q + [''] * (max_len - len(G_tfe.Q))
        events = G_tfe.Ye + [''] * (max_len - len(G_tfe.Ye))
        init_state = [G_tfe.q0] + [''] * (max_len - 1)
        
        df_g_tfe = pd.DataFrame({
            'States': states,
            'Events': events,
            'Initial State': init_state
        })
        df_g_tfe.to_excel(writer, sheet_name='G_tfe', index=False)
        
        if G_tfe.Delta:
            df_transitions = pd.DataFrame(G_tfe.Delta, columns=['From', 'Event', 'To'])
            df_transitions.to_excel(writer, sheet_name='G_tfe_transitions', index=False)
        
        # G_M
        max_len = max(len(G_M.Q), len(G_M.Ye), 1)
        states = G_M.Q + [''] * (max_len - len(G_M.Q))
        events = G_M.Ye + [''] * (max_len - len(G_M.Ye))
        init_state = [G_M.q0] + [''] * (max_len - 1)
        
        df_g_m = pd.DataFrame({
            'States': states,
            'Events': events,
            'Initial State': init_state
        })
        df_g_m.to_excel(writer, sheet_name='G_M', index=False)
        
        if G_M.Delta:
            df_transitions = pd.DataFrame(G_M.Delta, columns=['From', 'Event', 'To'])
            df_transitions.to_excel(writer, sheet_name='G_M_transitions', index=False)
        
        # CC
        # 确保CC中的状态没有重复
        cc_states = [str(state) for state in cc.Q]
        max_len = max(len(cc_states), len(cc.Ye), 1)
        states = cc_states + [''] * (max_len - len(cc_states))
        events = cc.Ye + [''] * (max_len - len(cc.Ye))
        init_state = [str(cc.q0)] + [''] * (max_len - 1)
        
        df_cc = pd.DataFrame({
            'States': states,
            'Events': events,
            'Initial State': init_state
        })
        df_cc.to_excel(writer, sheet_name='CC', index=False)
        
        if cc.Delta:
            # 确保转换没有重复
            transitions = []
            seen = set()
            for t in cc.Delta:
                transition_str = f"{str(t[0])}-{t[1]}-{str(t[2])}"
                if transition_str not in seen:
                    seen.add(transition_str)
                    transitions.append({'From': str(t[0]), 'Event': t[1], 'To': str(t[2])})
            
            df_transitions = pd.DataFrame(transitions)
            df_transitions.to_excel(writer, sheet_name='CC_transitions', index=False)
        
        # Merged Results
        max_len = max(len(result.Q), len(result.Ye), 1)
        states = result.Q + [''] * (max_len - len(result.Q))
        events = result.Ye + [''] * (max_len - len(result.Ye))
        init_state = [result.q0] + [''] * (max_len - 1)
        
        df_result = pd.DataFrame({
            'States': states,
            'Events': events,
            'Initial State': init_state
        })
        df_result.to_excel(writer, sheet_name='Merged_Results', index=False)
        
        if result.Delta:
            df_transitions = pd.DataFrame(result.Delta, columns=['From', 'Event', 'To'])
            df_transitions.to_excel(writer, sheet_name='Merged_transitions', index=False)
        
        # Observer
        max_len = max(len(Gobs.Q), len(Gobs.Ye), 1)
        states = [str(state) for state in Gobs.Q] + [''] * (max_len - len(Gobs.Q))
        events = Gobs.Ye + [''] * (max_len - len(Gobs.Ye))
        init_state = [str(Gobs.q0)] + [''] * (max_len - 1)
        
        df_obs = pd.DataFrame({
            'States': states,
            'Events': events,
            'Initial State': init_state
        })
        df_obs.to_excel(writer, sheet_name='Observer', index=False)
        
        if Gobs.Delta:
            df_transitions = pd.DataFrame(Gobs.Delta, columns=['From', 'Event', 'To'])
            df_transitions.to_excel(writer, sheet_name='Observer_transitions', index=False)

class G_tf:
    """G_tf结构体"""
    def __init__(self):
        self.X = ['x0', 'x1', 'x2']
        self.Y = [1, 2, 3]
        self.h = Container()
        self.h['x0'] = [1]
        self.h['x1'] = [1, 2]
        self.h['x2'] = [3]
        self.B = [['x0', 'x1'], ['x0', 'x2']]
        self.x0 = 'x0'
        self.y0 = 1
        self.B_f = [['x1', 'x0'], ['x1', 'x2']]
        self.X_tf = ['x1']
        self.ff = Container()
        self.ff['x1'] = [1, 2]
        self.FaultOccurrenceIntervals = Container()
        self.FaultOccurrenceIntervals['(x1,1)(x1,x0)'] = [1, 2]
        self.FaultOccurrenceIntervals['(x1,2)(x1,x0)'] = [1, 2]
        self.FaultOccurrenceIntervals['(x1,2)(x1,x2)'] = [2, 3]
def main():
    """主函数"""
    try:
        # 构造故障演化自动机和监视器
        print("Constructing timed fault evolution automaton...")
        G_tfe = construct_timed_fault_evolution_automaton(G_tf())
        
        print("Constructing fault monitor...")
        G_M = construct_fault_monitor(G_tfe)
        
        print("Computing concurrent composition...")
        cc = concurrent_composition(G_tfe, G_M)
        
        print("Merging CC states...")
        result = merge_cc_states(cc)
        
        print("Constructing observer...")
        Gobs = construct_observer(result)
        
        print("Saving results to Excel...")
        save_to_excel(G_tfe, G_M, cc, result, Gobs)
        print("Results have been saved to results.xlsx")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()