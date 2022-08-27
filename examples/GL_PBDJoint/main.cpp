#include <GlfwApp.h>

#include <SceneGraph.h>

#include <PBDRigidBody/PBDRigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include "Collision/NeighborElementQuery.h"

#include "PBDRigidBody/Joint.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<PBDRigidBodySystem<DataType3f>>());

	RigidBodyInfo rigidBody;
	BoxInfo box;

	box.center = Vec3f(0.5f,0.5f,0.5f);
	box.halfLength = Vec3f(0.4f, 0.05f, 0.05f);
	rigidBody.angularVelocity = Vec3f(0.0f,50.0f,0.0f);
	rigid->addBox(box, rigidBody);

	box.center = Vec3f(0.5f, 0.6f, 0.5f);
	box.halfLength = Vec3f(0.4f, 0.05f, 0.05f);
	rigidBody.angularVelocity = Vec3f(0.0f);
	box.rot = Quat<Real>(M_PI * 0.25f, Vec3f(0.0f,0.0f,1.0f));
	
	rigid->addBox(box, rigidBody);
	
	Joint<Real> j = Joint<Real>(
		0,1,
		0.0f,
		Vec3f( 0.0f,   0.0f, 1.0f),
		Vec3f(-0.4f,  -0.05f, 0.0f),
		Vec3f(-0.4f, -0.05f, 0.0f)
		);
	rigid->addJoint(j);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(0, 1, 1));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals
	auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
	rigid->stateTopology()->connect(elementQuery->inDiscreteElements());
	rigid->stateCollisionMask()->connect(elementQuery->inCollisionMask());
	rigid->graphicsPipeline()->pushModule(elementQuery);

	auto contactMapper = std::make_shared<ContactsToEdgeSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactMapper->inContacts());
	contactMapper->varScale()->setValue(0.02);
	rigid->graphicsPipeline()->pushModule(contactMapper);

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Vec3f(0, 1, 0));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);

	//Visualize contact points
	auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	rigid->graphicsPipeline()->pushModule(contactPointMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->setColor(Vec3f(1, 0, 0));
	contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pointRender);

	return scn;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(creatBricks());
	window.createWindow(1280, 768);
	window.mainLoop();

	return 0;
}


