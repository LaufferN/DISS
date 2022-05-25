from collections import defaultdict

import dfa
import funcy as fn
import pydot
from IPython.display import SVG
from IPython.display import HTML as html_print

from diss.concept_classes.dfa_concept import remove_stutter, DFAConcept
from diss.concept_classes.dfa_decomposition_concept import DFAProductConcept


COLOR_ALIAS = {
    'white': 'white',
    'yellow': '#ffff00', 
    'red': '#ff8b8b',
    'blue': '#afafff', 
    'green' : '#8ff45d'
}


# adapted from the dfa library
def get_dot(dfa_):
    dfa_dict, init = dfa.dfa2dict(dfa_)
    remove_stutter(dfa_dict)
    g = pydot.Dot(rankdir="LR")

    nodes = {}
    for i, (k, (v, _)) in enumerate(dfa_dict.items()):
        shape = "doublecircle" if v else "circle"
        nodes[k] = pydot.Node(i+1, label=f"{k}", shape=shape, color="white", fontcolor="white")
        g.add_node(nodes[k])

    edges = defaultdict(list)
    for start, (_, transitions) in dfa_dict.items():        
        for action, end in transitions.items():
            color = COLOR_ALIAS[str(action)]
            edges[start, end].append(color)
    
    init_node = pydot.Node(0, shape="point", label="", color="white")
    g.add_node(init_node)
    g.add_edge(pydot.Edge(init_node, nodes[init], color="white"))

    for (start, end), colors in edges.items():
        #color_list = f':'.join(colors)
        #g.add_edge(pydot.Edge(nodes[start], nodes[end], color=color_list))
        for color in colors:
            g.add_edge(pydot.Edge(nodes[start], nodes[end], label='◼', fontcolor=color, color="white"))
    g.set_bgcolor("#00000000")        
    return g

def view_dfas(dfas_or_concepts):
    if isinstance(dfas_or_concepts, DFAProductConcept):
        dfas_or_concepts = [c.dfa for c in dfas_or_concepts.dfa_concepts]
    svgs = [SVG(get_dot(dfa).create_svg()) for dfa in dfas_or_concepts]
    display(*svgs)

def view_dfa(dfa_or_concept):
    if isinstance(dfa_or_concept, DFAConcept):
        dfa_or_concept = dfa_or_concept.dfa
    pdot = get_dot(dfa_or_concept)
    display(SVG(pdot.create_svg()))


def tile(color='black'):
    color = COLOR_ALIAS.get(color, color)
    s = '&nbsp;'*4
    return f"<text style='border: solid 1px;background-color:{color}'>{s}</text>"


def ap_at_state(x, y, world):
    """Use sensor to create colored tile."""
    if (x, y) in world.overlay:
        color = world.overlay[(x,y)]

        if color in COLOR_ALIAS.keys():
            return tile(color)
    return tile('white')


def print_map(world):
    """Scan the board row by row and print colored tiles."""
    order = range(1, world.dim + 1)
    buffer = ''
    for y in order:
        chars = (ap_at_state(x, y, world) for x in order)
        buffer += '&nbsp;'.join(chars) + '<br>'
    display(html_print(buffer))


def print_trc(path, world, idx=0):
    actions = [s.action for s, kind in path if kind == 'env']
    states = [s for s, kind in path if kind == 'ego']
    obs = [ap_at_state(pos.x, pos.y, world) for pos in states]
    if len(actions) > len(obs):
        obs.append('')
    elif len(obs) > len(actions):
        actions.append('')
    trc = fn.interleave(obs, actions)

    display(html_print(f'trc {idx}:&nbsp;&nbsp;&nbsp;' + ''.join(trc)))
