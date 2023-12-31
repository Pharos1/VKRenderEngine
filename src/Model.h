#ifndef MODEL
#define MODEL

//#include <glad/glad.h> 
#include <GLAD/gl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SOIL2/stb_image.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Mesh.h"
#include "Shader.h"
#include "Texture.h"
#include "Vertex.h"
#include "Material.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;

class Model {
public:
	vector<Texture> textures_loaded;
	vector<Texture*> loadedTextures;
	vector<MaterialMesh>    meshes;
	string directory;

	//glm::mat4 localModelMat;

	Model(string const& path) {
		loadModel(path);
	}
	~Model() {
		for (int i = 0; i < textures_loaded.size(); i++) {
			delete& textures_loaded[i];
		}

		for (int i = 0; i < loadedTextures.size(); i++) {
			delete loadedTextures[i];
		}
	}
	Model() {};

	void Draw(Shader& shader) {
		for (unsigned int i = 0; i < meshes.size(); i++)
			meshes[i].Draw(shader);
	}
	void loadModel(string const& path) {
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph); //aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
			cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
			return;
		}
		directory = path.substr(0, max((int)path.find_last_of('/'), (int)path.find_last_of('\\')));

		processNode(scene->mRootNode, scene);
	}
	void processNode(aiNode* node, const aiScene* scene) {
		for (unsigned int i = 0; i < node->mNumMeshes; i++) {
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			meshes.push_back(processMesh(mesh, scene, node));
		}
		for (unsigned int i = 0; i < node->mNumChildren; i++) {
			processNode(node->mChildren[i], scene);
		}
	}
	MaterialMesh processMesh(aiMesh* mesh, const aiScene* scene, aiNode* node) {
		vector<Vertex> vertices;
		vector<unsigned int> indices;

		for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
			Vertex vertex;
			glm::vec3 vector;
			vector.x = mesh->mVertices[i].x;
			vector.y = mesh->mVertices[i].y;
			vector.z = mesh->mVertices[i].z;
			vertex.position = vector;

			if (mesh->HasNormals()) {
				vector.x = mesh->mNormals[i].x;
				vector.y = mesh->mNormals[i].y;
				vector.z = mesh->mNormals[i].z;
				vertex.normal = vector;
			}
			if (mesh->mTextureCoords[0]) {
				glm::vec2 vec;

				vec.x = mesh->mTextureCoords[0][i].x;
				vec.y = mesh->mTextureCoords[0][i].y;
				vertex.texCoord = vec;

				vector.x = mesh->mTangents[i].x;
				vector.y = mesh->mTangents[i].y;
				vector.z = mesh->mTangents[i].z;
				vertex.tangent = vector;
			}
			else
				vertex.texCoord = glm::vec2(0.0f, 0.0f);
			vertices.push_back(vertex);
		}
		for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
			aiFace face = mesh->mFaces[i];
			// retrieve all indices of the face and store them in the indices vector
			for (unsigned int j = 0; j < face.mNumIndices; j++)
				indices.push_back(face.mIndices[j]);
		}

		MaterialMesh finalMesh(vertices, indices);

		if (scene->HasMaterials()) {
			aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

			loadMaterial(finalMesh.material.albedo, material, aiTextureType_DIFFUSE);
			loadMaterial(finalMesh.material.normal, material, aiTextureType_HEIGHT);
			loadMaterial(finalMesh.material.metallic, material, aiTextureType_METALNESS);
			loadMaterial(finalMesh.material.roughness, material, aiTextureType_DIFFUSE_ROUGHNESS);
			loadMaterial(finalMesh.material.AO, material, aiTextureType_LIGHTMAP);

			finalMesh.material.initialized = true;
		}
		/*
		for (size_t row = 0; row < 4; row++) {
			for (size_t col = 0; col < 4; col++) {
				localModelMat[row][col] = node->mTransformation[row][col];
			}
		}
		*/

		return finalMesh;
	}
	void loadMaterial(Texture& materialTexture, aiMaterial* material, aiTextureType type) {
		aiString texturePath;
		if (material->GetTexture(type, 0, &texturePath) == -1) return;

		std::string path = this->directory + "/" + texturePath.C_Str();

		bool skip = false;

		for (int i = 0; i < textures_loaded.size(); i++) {
			if (textures_loaded[i].path == path) {
				skip = true;

				materialTexture = textures_loaded[i];

				break;
			}
		}
		if (!skip) { // && std::strcmp(texturePath.C_Str(), "") != 0
			materialTexture.loadTexture(path);
			textures_loaded.push_back(materialTexture);
		}
	}
};
#endif
