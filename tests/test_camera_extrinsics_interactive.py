import math

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from main import CameraExtrinsics
import numpy

app = dash.Dash(__name__)
points_3d = numpy.asarray([
	[0, 0, 0, 1],
	[1, 0, 0, 1],
	[0, 1, 0, 1],
	[0, 0, 1, 1],
])

app.layout = html.Div([
	html.Div([
		dcc.Graph(id='points-3d'),
		dcc.Graph(id='projected-2d'),
	]),
	html.Div([
		"Camera Translation (XYZ): ",
		#dcc.Input(id='camera-x', value='0', type='number'),
		dcc.Slider(id='camera-x', min=-10, max=10, value=0, step=0.5),
		dcc.Slider(id='camera-y', min=-10, max=10, value=0, step=0.5),
		dcc.Slider(id='camera-z', min=-10, max=10, value=1, step=0.5),
		"Camera Rotation (XYZ): ",
		dcc.Slider(id='camera-rx', min=-180, max=180, value=0, step=10),
		dcc.Slider(id='camera-ry', min=-180, max=180, value=0, step=10),
		dcc.Slider(id='camera-rz', min=-180, max=180, value=0, step=10),
	]),

])

@app.callback(
	Output(component_id='points-3d', component_property='figure'),
	Output(component_id='projected-2d', component_property='figure'), # children, figure, or data
	Input(component_id='camera-x', component_property='value'),
	Input(component_id='camera-y', component_property='value'),
	Input(component_id='camera-z', component_property='value'),
	Input(component_id='camera-rx', component_property='value'),
	Input(component_id='camera-ry', component_property='value'),
	Input(component_id='camera-rz', component_property='value'),
)
def update_camera_projection(x, y, z, rx, ry, rz):
	rx = math.radians(rx)
	ry = math.radians(ry)
	rz = math.radians(rz)

	pts = CameraExtrinsics(
			x_rotation=rx, y_rotation=ry, z_rotation=rz,
			x_translation=x, y_translation=y, z_translation=z) \
		.project_points(points_3d, renormalize=True)
	fig3d = go.Figure(data=[
		#go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers'),
		# Points 3D:
		go.Scatter3d(x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2], mode='markers'),
		# Camera origin:
		go.Scatter3d(x=[x], y=[y], z=[z], mode='markers'),
		# Camera Target:
		go.Scatter3d(x=[x+math.cos(rx)], y=[y+math.cos(ry)], z=[z+math.sin(rz)], mode='markers')
	])
	fig2d = go.Figure(
		data=[go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='markers'),],
		layout={
			"xaxis.uirevision": True,
			"yaxis.uirevision": True,
		}
	)
	print(pts)
	return fig3d, fig2d
	#return 'Output: {}'.format(input_value)


if __name__ == '__main__':
	app.run_server(debug=True)