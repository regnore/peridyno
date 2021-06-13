#include "GlfwGUI/GlfwApp.h"

#include "Framework/SceneGraph.h"
#include "Framework/Log.h"

#include "Peridynamics/ParticleElasticBody.h"
#include "Peridynamics/Cloth.h"

#include "ParticleSystem/StaticBoundary.h"

#include "RigidBody/RigidBody.h"

#include "module/PointRender.h"
#include "module/SurfaceRender.h"

using namespace std;
using namespace dyno;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vec3f(1.5f, 3.0f, 1.5f));
	scene.setLowerBound(Vec3f(-1.5f, 0.0f, 1.5f));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadSDF("../../data/t-shirt/body-scaled.sdf");
// 	root->loadCube(Vec3f(-1.5, 0, -1.5), Vec3f(1.5, 3.5, 1.5), 0.05f, true);
// 	root->loadShpere(Vec3f(0.0, 0.7f, 0.0), 0.15f, 0.005f, false, true);

	std::shared_ptr<Cloth<DataType3f>> child3 = std::make_shared<Cloth<DataType3f>>();
	root->addParticleSystem(child3);

	auto clothRender = std::make_shared<SurfaceRenderer>();
	clothRender->setColor(glm::vec3(1.0, 1.0, 1.0));
	child3->getSurface()->addVisualModule(clothRender);

	auto m_pointsRender = std::make_shared<PointRenderer>();
	m_pointsRender->setColor(glm::vec3(1, 1, 1));
	child3->addVisualModule(m_pointsRender);
	child3->setVisible(false);

//	child3->setMass(1.0);
//  child3->loadParticles("../../data/cloth/cloth.obj");
//  child3->loadSurface("../../data/cloth/cloth.obj");

	child3->loadParticles("../../data/t-shirt/t-shirt.obj");
	child3->loadSurface("../../data/t-shirt/t-shirt.obj");

	//child3->scale(0.5);

	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	root->addRigidBody(rigidbody);
	rigidbody->loadShape("../../data/t-shirt/body-scaled.obj");
	rigidbody->scale(0.95);
	rigidbody->translate(Vec3f(0, 0.07, 0));
	rigidbody->setActive(false);

	auto bodyRender = std::make_shared<SurfaceRenderer>();
	bodyRender->setColor(glm::vec3(0.1, 1.0, 1));
	rigidbody->getSurface()->addVisualModule(bodyRender);
}

int main()
{
	CreateScene();

	GlfwApp window;
	window.createWindow(1024, 768);

	window.mainLoop();
	return 0;
}


