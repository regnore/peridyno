#include <QtApp.h>

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

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatBricks()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto rigid = scn->addNode(std::make_shared<PBDRigidBodySystem<DataType3f>>());
	SphereInfo sphere;
	RigidBodyInfo rigidSphere;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < i; j++) 
		{
			sphere.center = Vec3f(2*0.025f * i, 0.025f, 2*0.025f * j);
			sphere.radius = 0.025f;
			rigid->addSphere(sphere, rigidSphere);
		}
	}
	rigidSphere.linearVelocity = Vec3f(9.0f, 0.0f, -6.0f);
	sphere.center = Vec3f(-0.2f, 0.025f,0.3f);
	sphere.radius = 0.025f;
	rigid->addSphere(sphere, rigidSphere);

	/*BoxInfo box;
	RigidBodyInfo rigidBox;
	box.center = Vec3f(0.1f,0.1f,0.1f);
	box.halfLength = Vec3f(0.1f);
	rigid->addBox(box, rigidBox);*/

	/*SphereInfo sphere;
	RigidBodyInfo rigidSphere;
	rigidSphere.linearVelocity=Vec3f(0.0f,-5.0f,0.0f);
	sphere.center = Vec3f(0.1f, 0.2f, 0.1f);
	sphere.radius = 0.1f;
	rigid->addSphere(sphere, rigidSphere);
	sphere.center = Vec3f(0.1f, 1.5f, 0.1f);
	rigidSphere.linearVelocity = Vec3f(0.0f, 0.0f, 0.0f);
	sphere.radius = 1.0f;
	rigid->addSphere(sphere, rigidSphere);*/

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
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
	QtApp window;
	window.setSceneGraph(creatBricks());
	window.createWindow(1280, 768);
	window.mainLoop();

	return 0;
}


