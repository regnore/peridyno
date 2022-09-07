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
	//rigid->varDynamicFrictionEnabled()->setValue(false);
	//rigid->varStaticFrictionEnabled()->setValue(false);
	RigidBodyInfo rigidBody;
	BoxInfo box;

	int n = 1000;
	box.halfLength = Vec3f(0.4f, 0.05f, 0.05f)/100.0f;
	box.center = Vec3f(0.5f, 0.4f, 0.5f) / 100.0f;
	//rigidBody.angularVelocity = Vec3f(0.0f, 500.0f, 0.0f);
	box.rot = Quat<Real>(M_PI * 1.0f / 2.0f, Vec3f(0.0f, 0.0f, 1.0f).normalize());
	rigid->addBox(box, rigidBody);
	for (int i = 0; i < n; i++)
	{
		int r = i % 4;
		rigid->addBox(box, rigidBody);
		Joint<Real> j = Joint<Real>(i, i + 1);
		switch (r) 
		{
			case 0:
				j.setHinge(
					Vec3f(0.40f, -0.051f, 0.0f) / 100.0f,
					Vec3f(0.40f, 0.051f, 0.0f) / 100.0f,
					M_PI);
				rigid->addJoint(j);
				break;
			case 1:
				j.setHinge(
					Vec3f(-0.40f, -0.051f, 0.0f) / 100.0f,
					Vec3f(0.40f, -0.051f, 0.0f) / 100.0f,
					0.0f);
				rigid->addJoint(j);
				break;
			case 2:
				j.setHinge(
					Vec3f(-0.40f, 0.051f, 0.0f) / 100.0f,
					Vec3f(-0.40f, -0.051f, 0.0f) / 100.0f,
					M_PI);
				rigid->addJoint(j);
				break;
			case 3:
				j.setHinge(
					Vec3f(0.40f, 0.051f, 0.0f) / 100.0f,
					Vec3f(-0.40f, 0.051f, 0.0f) / 100.0f,
					0.0f);
				rigid->addJoint(j);
				break;
		}
	}

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
	pRender->varPointSize()->setValue(0.0002f);
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

	/*auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Vec3f(0, 1, 0));
	contactMapper->outEdgeSet()->connect(wireRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(wireRender);*/

	////Visualize contact points
	//auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
	//elementQuery->outContacts()->connect(contactPointMapper->inContacts());
	//rigid->graphicsPipeline()->pushModule(contactPointMapper);

	//auto pointRender = std::make_shared<GLPointVisualModule>();
	//pointRender->setColor(Vec3f(1, 0, 0));
	//contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
	//rigid->graphicsPipeline()->pushModule(pointRender);

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


