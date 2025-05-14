import src.turtlebot2_api
from src.AStar_path_finding import TurtleBotAStar
import flask

app = flask.Flask(__name__)


x,y = app.route('/move/<int:x>/<int:y>')

turtle_bot = TurtleBotAStar()
# is_object -> bool -> object on the way
# distance_object -> int -> distance to center obj
# objs -> list -> avarage distance of 16 squares on pictures
# final -> ? bool
turtle_bot.move(x,y)


