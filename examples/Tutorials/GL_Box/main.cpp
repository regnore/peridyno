#include <GlfwApp.h>
#include <GLRenderEngine.h>

#include "SceneGraph.h"
#include "GLSurfaceVisualModule.h"
#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

using namespace dyno;

template<typename TDataType>
class Box : public Node
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TopologyModule::Triangle Triangle;

	Box() 
	{
		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		std::vector<Coord> points;
		points.push_back(Coord(-0.5f, 0.0f, -0.5f));
		points.push_back(Coord(0.5f, 0.0f, -0.5f));
		points.push_back(Coord(0.5f, 0.0f, 0.5f));
		points.push_back(Coord(-0.5f, 0.0f, 0.5f));
		points.push_back(Coord(-0.5f, 1.0f, -0.5f));
		points.push_back(Coord(0.5f, 1.0f, -0.5f));
		points.push_back(Coord(0.5f, 1.0f, 0.5f));
		points.push_back(Coord(-0.5f, 1.0f, 0.5f));
		triSet->setPoints(points);

		std::vector<Triangle> tris;
		tris.push_back(Triangle(0, 1, 2));
		tris.push_back(Triangle(0, 3, 2));
		tris.push_back(Triangle(0, 4, 5));
		tris.push_back(Triangle(0, 1, 5));
		tris.push_back(Triangle(0, 3, 7));
		tris.push_back(Triangle(0, 4, 7));
		tris.push_back(Triangle(6, 1, 2));
		tris.push_back(Triangle(6, 3, 2));
		tris.push_back(Triangle(6, 4, 5));
		tris.push_back(Triangle(6, 1, 5));
		tris.push_back(Triangle(6, 3, 7));
		tris.push_back(Triangle(6, 4, 7));
		triSet->setTriangles(tris);

		triSet->updateEdges();
		triSet->updateVertexNormal();

		this->stateTopology()->setDataPtr(triSet);
	};

	DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
};

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto instanceNode = scn->addNode(std::make_shared<Box<DataType3f>>());

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Vec3f(1.0f, 0.0f, 0.0f));
	pointRender->varPointSize()->setValue(0.02f);
	instanceNode->stateTopology()->connect(pointRender->inPointSet());
	instanceNode->graphicsPipeline()->pushModule(pointRender);

	auto edgeRender = std::make_shared<GLWireframeVisualModule>();
	edgeRender->setColor(Vec3f(0.0f,0.0f,0.0f));
	instanceNode->stateTopology()->connect(edgeRender->inEdgeSet());
	instanceNode->graphicsPipeline()->pushModule(edgeRender);

	auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
	surfaceRender->setColor(Vec3f(0.8f, 0.52f, 0.25f));
	instanceNode->stateTopology()->connect(surfaceRender->inTriangleSet());
	instanceNode->graphicsPipeline()->pushModule(surfaceRender);

	scn->setUpperBound({ 4, 4, 4 });

	GlfwApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}
