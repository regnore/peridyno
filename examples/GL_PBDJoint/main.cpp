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

	box.center = Vec3f(0.5f,0.7f,0.5f);
	box.halfLength = Vec3f(0.4f, 0.04f, 0.04f);
	//box.rot= Quat<Real>(M_PI /4.0f, Vec3f(1.0f, 0.0f, 0.0f));
	//rigidBody.angularVelocity = Vec3f(0.0f, 50.0f, 50.0f);
	box.rot = Quat<Real>(M_PI * 1.0f / 3.0f, Vec3f(0.0f, 0.0f, 1.0f).normalize());
	rigidBody.angularVelocity = Vec3f(0.0f, 100.0f,0.0f);
	rigid->addBox(box, rigidBody);

	box.rot = Quat<Real>(M_PI * 5.0f / 6.0f, Vec3f(0.0f,0.0f,1.0f).normalize());

	rigid->addBox(box, rigidBody);

	box.rot = Quat<Real>(M_PI * 1.0f / 3.0f, Vec3f(0.0f, 0.0f, 1.0f).normalize());
	 
	rigid->addBox(box, rigidBody);

	box.rot = Quat<Real>(-M_PI * 1.0f / 6.0f, Vec3f(0.0f, 0.0f, 1.0f).normalize());

	rigid->addBox(box, rigidBody);
	
	Joint<Real> j1 = Joint<Real>(
		0,1,
		0.0f,
		Vec3f( 0.0f,   0.0f, 1.0f),
		Vec3f(-0.40f,  0.041f, 0.0f),
		Vec3f(-0.40f, -0.041f, 0.0f)
		);
	rigid->addJoint(j1);

	Joint<Real> j2 = Joint<Real>(
		1, 2,
		0.0f,
		Vec3f(0.0f, 0.0f, 1.0f),
		Vec3f(0.40f, -0.041f, 0.0f),
		Vec3f(-0.40f, -0.041f, 0.0f)
		);
	rigid->addJoint(j2);

	Joint<Real> j3 = Joint<Real>(
		2, 3,
		0.0f,
		Vec3f(0.0f, 0.0f, 1.0f),
		Vec3f(0.40f, -0.041f, 0.0f),
		Vec3f(-0.40f, -0.041f, 0.0f)
		);
	rigid->addJoint(j3);

	Joint<Real> j4 = Joint<Real>(
		3, 0,
		0.0f,
		Vec3f(0.0f, 0.0f, 1.0f),
		Vec3f(0.40f, -0.041f, 0.0f),
		Vec3f(0.40f, 0.041f, 0.0f)
		);
	rigid->addJoint(j4);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(0, 1, 1));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	auto eRender = std::make_shared<GLWireframeVisualModule>();
	eRender->setColor(Vec3f(0, 0, 0));
	mapper->outTriangleSet()->connect(eRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(eRender);

	auto pRender = std::make_shared<GLPointVisualModule>();
	pRender->setColor(Vec3f(0, 0, 0));
	mapper->outTriangleSet()->connect(pRender->inPointSet());
	rigid->graphicsPipeline()->pushModule(pRender);

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


