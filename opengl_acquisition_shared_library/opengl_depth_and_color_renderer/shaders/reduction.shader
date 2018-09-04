//FROM https://www.shadertoy.com/view/lstXzs

#define STARTING_MODULUS 512

bool reduceHere(int mm, vec2 c) {
	vec2 m = floor(mod(c, float(mm)));
    return all(equal(m, vec2(0.0)));
}

vec4 T(sampler2D ch, vec2 uv) {
	return clamp(uv, 0.0, 1.0) == uv ? texture(ch, uv) : vec4(0.0);   
}

const ivec4 wi = ivec4(1, 2, 4, 8);

vec4 reduce(sampler2D ch0, sampler2D ch1, int mi, vec2 c) {
    vec4 r = vec4(0.0);
    for (int p = 0; p < 4; p++) {
        // this ought to be done with bitwise ops but they are not supported here
        float m = float(wi[p] * mi/2);
        int n = wi[p] * mi;
        if (reduceHere(n, c)) {
            vec3 i = vec3(m, m, 0.0) / iResolution.xyy;
            vec2 uv = c / iResolution.xy;
            vec2 uv_e  = uv + i.xz;
            vec2 uv_ne = uv + i.xy;
            vec2 uv_n  = uv + i.zy;
            if (p == 0) {
                r[p] = T(ch0, uv).w + T(ch0, uv_e).w + T(ch0, uv_ne).w + T(ch0, uv_n).w;
            } else {
                r[p] = T(ch1, uv)[p-1] + T(ch1, uv_e)[p-1] + T(ch1, uv_ne)[p-1] + T(ch1, uv_n)[p-1];
            }
        } else {
           r[p] = 0.0; 
        }
    }
    return r;
}

void mainImage( out vec4 O, in vec2 U )
{
	O = vec4(reduce(iChannel0, iChannel1, STARTING_MODULUS, U));
}
