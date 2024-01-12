#include <stdio.h>
#include <arm_sve.h>

int main()
{
	const int len = 16;
	uint32_t *a = new uint32_t[len];
	uint8_t *b = new uint8_t[len];
	uint32_t *c = new uint32_t[len];
	for (int i = 0; i < len; i++)
	{
		a[i] = i;
	}
	a[11] = 250;

	// svbool_t pg0 = svptrue_b32();
	svbool_t pg0 = svwhilelt_b32(0, 13);
	svuint32_t v0 = svld1_u32(pg0, a);

	printf("let's test the truncation of svuint32_t to svuint8_t\n");
	svst1b_u32(pg0, b, v0);
	for (int i = 0; i < len; i++)
	{
		printf("%u\n", b[i]);
	}

	printf("let's test the extension of svuint8_t to svuint32_t\n");
	svuint32_t v1 = svld1ub_u32(pg0, b);
	svst1_u32(pg0, c, v1);
	for (int i = 0; i < len; i++)
	{
		printf("%u\n", c[i]);
	}

	delete[] a;
	delete[] b;
	delete[] c;
	return 0;
}
