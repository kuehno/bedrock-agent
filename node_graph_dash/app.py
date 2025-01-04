import plotly.graph_objects as go
import networkx as nx
import dash_dangerously_set_inner_html
import json

with open('agent_graph.json') as f:
    agent_graph = json.load(f)

# Recursively extract agent relationships
def extract_agent_relationships(agent_name, agent_data, relationships, messages):
    for handoff_agent_name, handoff_agent_data in agent_data.get('HandoffAgents', {}).items():
        relationships.append((agent_name, handoff_agent_name))
        extract_agent_relationships(handoff_agent_name, handoff_agent_data, relationships, messages)
    messages[agent_name] = agent_data.get('messages', [])

relationships = []
messages = {}
extract_agent_relationships('ExpertAgent', agent_graph['ExpertAgent'], relationships, messages)

G = nx.DiGraph()

for agent, handoff_agent in relationships:
    G.add_node(agent)
    G.add_node(handoff_agent)
    G.add_edge(agent, handoff_agent)

pos = nx.spring_layout(G, k=1)

edge_trace = go.Scatter(
    x=(),
    y=(),
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)

node_trace = go.Scatter(
    x=(),
    y=(),
    text=(),
    mode='markers+text',
    hoverinfo='text',
    textfont=dict(size=10, color='white'),  # Reduced font size and white color
    marker=dict(
        showscale=False,
        color=(),
        size=120,
        line_width=2
    )
)

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)
    node_trace['text'] += (node,)
    node_trace['marker']['color'] += ('#1f78b4',)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                ))

fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='agent-graph', figure=fig, style={
            'flex': '1', 
            'width': '50%', 
            'height': '100vh',
            'padding-right': '10px'
            }
        ),
        html.Div(id='chat-messages', style={
            'whiteSpace': 'pre-wrap', 
            'verticalAlign': 'top', 
            'flex': '1', 
            'height': '100vh', 
            'overflowY': 'scroll', 
            'border': '1px solid black', 
            'padding': '10px',
            'width': '50%'
            }
        )
    ], style={
        'display': 'flex', 
        'flexDirection': 'row', 
        'width': '100%', 
        'height': '100vh',
        'gap': '10px'
        }
    )
])

@app.callback(
    Output('chat-messages', 'children'),
    [Input('agent-graph', 'clickData')]
)
def display_messages(clickData):
    if clickData is None:
        return "Click on a node to see the chat messages."
    
    node = clickData['points'][0]['text']
    messages_text = ""
    for msg in messages[node]:
        role = msg['role'].upper()
        content = msg['content']
        for c in content:
            if 'text' in c:
                css_class = 'user-message' if role == 'USER' else 'assistant-message' if role == 'ASSISTANT' else ''
                messages_text += f"<span class='{css_class}'>{role}: {c.get('text', 'No text available')}</span><br>"
            if 'toolUse' in c:
                tool_name = c['toolUse']['name']
                tool_input = c['toolUse']['input']
                messages_text += f"<span class='tool-called'>TOOL CALLED: {tool_name} with input {tool_input}</span><br>"
            if 'toolResult' in c:
                tool_result = c['toolResult']['content'][0].get('text', 'No result available')
                messages_text += f"<span class='tool-result'>TOOL RESULT: {tool_result}</span><br>"
            messages_text += "<br>"
    return html.Div([dash_dangerously_set_inner_html.DangerouslySetInnerHTML(messages_text)])

if __name__ == '__main__':
    app.run_server(debug=True)