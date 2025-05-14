import src.turtlebot2_api
from src.AStar_path_finding import TurtleBotAStar
import flask

app = flask.Flask(__name__)


x,y = app.route('/move/<int:x>/<int:y>')

turtle_bot = TurtleBotAStar()

turtle_bot.move(x,y)


